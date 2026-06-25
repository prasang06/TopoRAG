import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from typing import Tuple, Dict, Any, Callable

# Enable double precision or keep default float32 (float32 is highly performant and standard)
jax.config.update("jax_enable_x64", False)

class GCNLayer(nn.Module):
    """
    Custom Graph Convolutional Network (GCN) Layer implemented from scratch using JAX.
    
    Propagation rule:
      H^(l+1) = activation( D^(-1/2) * A_tilde * D^(-1/2) * H^(l) * W^(l) )
    """
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, h: jnp.ndarray, norm_adj: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            h: Node feature matrix of shape (N, in_features).
            norm_adj: Symmetric normalized adjacency matrix of shape (N, N).
        Returns:
            Propagated node representation of shape (N, features).
        """
        in_features = h.shape[-1]
        
        # Initialize weight matrix W using Glorot/Xavier uniform initialization
        w = self.param(
            'weights',
            nn.initializers.glorot_uniform(),
            (in_features, self.features)
        )
        
        # Feature transformation (linear mapping: H * W)
        hw = jnp.matmul(h, w)
        
        # Spatial graph propagation: L_sym * (H * W)
        # norm_adj represents D^(-1/2) * (A + I) * D^(-1/2)
        out = jnp.matmul(norm_adj, hw)
        
        # Add bias if requested
        if self.use_bias:
            b = self.param(
                'bias',
                nn.initializers.zeros_init(),
                (self.features,)
            )
            out = out + b
            
        return out


class GCN(nn.Module):
    """
    A 2-layer Graph Convolutional Network for learning node representations.
    """
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, h: jnp.ndarray, norm_adj: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            h: Node feature matrix of shape (N, in_features).
            norm_adj: Symmetric normalized adjacency matrix of shape (N, N).
        Returns:
            Refined structural embeddings of shape (N, out_dim).
        """
        # Layer 1: Convolution + ReLU Activation
        h = GCNLayer(features=self.hidden_dim)(h, norm_adj)
        h = nn.relu(h)
        
        # Layer 2: Convolution (Output Layer, returns dense latent embeddings)
        h = GCNLayer(features=self.out_dim)(h, norm_adj)
        return h


def compute_normalized_adjacency(adj: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the symmetric normalized adjacency matrix programmatically:
      A_tilde = A + I
      D_ii = sum_j (A_tilde_ij)
      A_norm = D^(-1/2) * A_tilde * D^(-1/2)
    """
    N = adj.shape[0]
    
    # 1. Programmatic addition of self-loops (A_tilde = A + I)
    A_tilde = adj + jnp.eye(N)
    
    # 2. Calculate row/column degrees
    d = jnp.sum(A_tilde, axis=1)
    
    # 3. Compute D^(-1/2) with safety checks for zero degrees (though self-loops guarantee degree >= 1)
    d_inv_sqrt = jnp.power(d, -0.5)
    d_inv_sqrt = jnp.where(jnp.isinf(d_inv_sqrt) | jnp.isnan(d_inv_sqrt), 0.0, d_inv_sqrt)
    
    # 4. Multiply row-wise and column-wise to avoid constructing a large diagonal matrix:
    # A_norm_ij = A_tilde_ij * d_inv_sqrt_i * d_inv_sqrt_j
    # This is more memory-efficient and equivalent to D^(-1/2) @ A_tilde @ D^(-1/2)
    A_norm = A_tilde * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    
    return A_norm


def train_gcn_unsupervised(
    X: np.ndarray,
    A: np.ndarray,
    hidden_dim: int = 64,
    out_dim: int = 32,
    epochs: int = 150,
    lr: float = 0.01,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Trains a 2-layer GCN using self-supervised Link Prediction.
    
    The objective is to force papers that cite each other or share direct structure
    to map closer in the GCN embedding space.
    
    Args:
        X: Node features of shape (N, D).
        A: Binary adjacency matrix of shape (N, N).
        hidden_dim: Number of hidden units in the first layer.
        out_dim: Output dimension of representation.
        epochs: Number of optimization steps.
        lr: Adam optimizer learning rate.
        seed: PRNG random seed.
        
    Returns:
        embeddings: Trained node embeddings of shape (N, out_dim).
        params: Trained Flax parameter dictionary.
        metrics: History of training loss metrics.
    """
    N = X.shape[0]
    if N == 0:
        return np.empty((0, out_dim)), {"loss": []}

    # Convert to JAX arrays
    jax_X = jnp.array(X, dtype=jnp.float32)
    jax_A = jnp.array(A, dtype=jnp.float32)
    
    # Precompute symmetric normalized adjacency matrix
    A_norm = compute_normalized_adjacency(jax_A)

    # Initialize GCN Model and parameters
    model = GCN(hidden_dim=hidden_dim, out_dim=out_dim)
    key = jax.random.PRNGKey(seed)
    init_params = model.init(key, jax_X, A_norm)['params']

    # Initialize Optax optimizer
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(init_params)

    # We use a weighted binary cross-entropy loss to address sparsity of the graph
    num_pos = jnp.sum(jax_A)
    num_neg = jax_A.size - num_pos
    pos_weight = num_neg / jnp.maximum(num_pos, 1.0)

    @jax.jit
    def loss_fn(params: Any) -> jnp.ndarray:
        # Forward pass to get node embeddings H
        embeddings = model.apply({'params': params}, jax_X, A_norm)
        # Inner-product decoder to reconstruct adjacency matrix logits
        logits_pred = jnp.matmul(embeddings, embeddings.T)
        
        # Stable binary cross-entropy loss calculation
        max_val = jnp.clip(logits_pred, 0, None)
        loss = max_val - logits_pred * jax_A + jnp.log(1.0 + jnp.exp(-jnp.abs(logits_pred)))
        
        # Apply pos_weight for positive edges to balance gradients
        weight_mask = jnp.where(jax_A == 1.0, pos_weight, 1.0)
        weighted_loss = loss * weight_mask
        
        return jnp.mean(weighted_loss)

    @jax.jit
    def step_fn(params: Any, opt_state: Any) -> Tuple[Any, Any, jnp.ndarray]:
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = tx.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss_val

    # Training Loop
    params = init_params
    loss_history = []
    
    for epoch in range(epochs):
        params, opt_state, loss_val = step_fn(params, opt_state)
        loss_history.append(float(loss_val))
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"GCN Self-Supervised Training - Epoch {epoch+1:03d}/{epochs:03d} | Loss: {loss_val:.4f}")

    # Compute final trained embeddings
    final_embeddings = model.apply({'params': params}, jax_X, A_norm)
    
    return np.array(final_embeddings), params, {"loss": loss_history}

