import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn
import optax
import numpy as np
from typing import Tuple, Dict, Any, List

# Enable double precision or keep default float32 (float32 is highly performant and standard)
jax.config.update("jax_enable_x64", False)

class MultiHeadGATLayer(nn.Module):
    out_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        head_dim = self.out_dim // self.num_heads
        
        head_outputs = []
        for h in range(self.num_heads):
            attention_query_fn = nn.Dense(head_dim, name=f"query_{h}")
            @jraph.concatenated_args
            def attention_logit_fn(features):
                return nn.leaky_relu(nn.Dense(1, name=f"logit_{h}")(features))
                
            gn = jraph.GAT(attention_query_fn, attention_logit_fn)
            head_outputs.append(gn(graph).nodes)
            
        multi_head_nodes = jnp.concatenate(head_outputs, axis=-1)
        
        # Residual connection
        if graph.nodes.shape[-1] == multi_head_nodes.shape[-1]:
            multi_head_nodes = multi_head_nodes + graph.nodes
        else:
            residual = nn.Dense(multi_head_nodes.shape[-1], name="residual")(graph.nodes)
            multi_head_nodes = multi_head_nodes + residual
            
        return graph._replace(nodes=multi_head_nodes)

class GAT(nn.Module):
    """
    A 2-layer Graph Attention Network with Multi-Head Attention, Dropout, and Residuals.
    """
    hidden_dim: int
    out_dim: int
    num_heads: int = 4
    dropout_rate: float = 0.6

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, deterministic: bool = False) -> jraph.GraphsTuple:
        # Layer 1
        graph = graph._replace(nodes=nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(graph.nodes))
        graph = MultiHeadGATLayer(self.hidden_dim, self.num_heads, name="layer1")(graph)
        graph = graph._replace(nodes=nn.elu(graph.nodes))
        
        # Layer 2
        graph = graph._replace(nodes=nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(graph.nodes))
        # The output layer typically uses 1 head to collapse to the final embedding dimension
        graph = MultiHeadGATLayer(self.out_dim, 1, name="layer2")(graph)
        
        return graph

def dense_to_jraph(X: np.ndarray, A: np.ndarray) -> jraph.GraphsTuple:
    """
    Converts dense feature matrix X and adjacency matrix A into a jraph.GraphsTuple.
    Ensures self-loops are added as GAT expects them for proper attention over the node itself.
    """
    N = X.shape[0]
    
    # Add self loops
    A_tilde = A + np.eye(N)
    A_tilde = np.clip(A_tilde, 0, 1) # Ensure binary
    
    # Get edge indices
    senders, receivers = np.nonzero(A_tilde)
    
    return jraph.GraphsTuple(
        nodes=jnp.array(X, dtype=jnp.float32),
        edges=None,
        senders=jnp.array(senders, dtype=jnp.int32),
        receivers=jnp.array(receivers, dtype=jnp.int32),
        n_node=jnp.array([N]),
        n_edge=jnp.array([len(senders)]),
        globals=None
    )

def train_gat_unsupervised(
    X: np.ndarray,
    A: np.ndarray,
    hidden_dim: int = 64,
    out_dim: int = 32,
    epochs: int = 150,
    lr: float = 0.01,
    seed: int = 42,
    num_heads: int = 4,
    dropout_rate: float = 0.6
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Trains a 2-layer optimized GAT using self-supervised Link Prediction.
    """
    N = X.shape[0]
    if N == 0:
        return np.empty((0, out_dim)), {}, {"loss": []}

    # Prepare jraph Graph
    graph = dense_to_jraph(X, A)
    jax_A = jnp.array(A, dtype=jnp.float32)

    # Initialize GAT Model and parameters
    model = GAT(hidden_dim=hidden_dim, out_dim=out_dim, num_heads=num_heads, dropout_rate=dropout_rate)
    key = jax.random.PRNGKey(seed)
    key, init_dropout_key = jax.random.split(key)
    init_params = model.init({'params': key, 'dropout': init_dropout_key}, graph, deterministic=False)['params']

    # Initialize Optax optimizer
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(init_params)

    # We use a weighted binary cross-entropy loss to address sparsity of the graph
    num_pos = jnp.sum(jax_A)
    num_neg = jax_A.size - num_pos
    pos_weight = num_neg / jnp.maximum(num_pos, 1.0)

    @jax.jit
    def loss_fn(params: Any, dropout_key: Any) -> jnp.ndarray:
        # Forward pass (training mode, dropout enabled)
        out_graph = model.apply({'params': params}, graph, deterministic=False, rngs={'dropout': dropout_key})
        embeddings = out_graph.nodes
        
        # Inner-product decoder
        logits_pred = jnp.matmul(embeddings, embeddings.T)
        
        # Stable binary cross-entropy
        max_val = jnp.clip(logits_pred, 0, None)
        loss = max_val - logits_pred * jax_A + jnp.log(1.0 + jnp.exp(-jnp.abs(logits_pred)))
        
        weight_mask = jnp.where(jax_A == 1.0, pos_weight, 1.0)
        weighted_loss = loss * weight_mask
        
        return jnp.mean(weighted_loss)

    @jax.jit
    def step_fn(params: Any, opt_state: Any, dropout_key: Any) -> Tuple[Any, Any, jnp.ndarray]:
        loss_val, grads = jax.value_and_grad(loss_fn)(params, dropout_key)
        updates, next_opt_state = tx.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss_val

    # Training Loop
    params = init_params
    loss_history = []
    
    # Random key for dropout per step
    step_key = key
    
    for epoch in range(epochs):
        step_key, dropout_key = jax.random.split(step_key)
        params, opt_state, loss_val = step_fn(params, opt_state, dropout_key)
        loss_history.append(float(loss_val))
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"GAT Self-Supervised Training - Epoch {epoch+1:03d}/{epochs:03d} | Loss: {loss_val:.4f}")

    # Compute final trained embeddings deterministically (no dropout)
    final_graph = model.apply({'params': params}, graph, deterministic=True)
    final_embeddings = final_graph.nodes
    
    return np.array(final_embeddings), params, {"loss": loss_history}
