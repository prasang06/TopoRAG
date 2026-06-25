import networkx as nx
import numpy as np
import json
import math
from collections import Counter
from typing import Dict, Any, List, Tuple, Set

class GraphRAGRetrievalEngine:
    """
    Topology-aware Graph-RAG Retrieval Engine that uses NetworkX for structural analysis
    and merges semantic similarity with topological network centrality.
    """
    def __init__(self, alpha: float = 0.4):
        """
        Args:
            alpha: Weighting factor for topology-based boosting in hybrid ranking.
                   Combined Score = Semantic Similarity + alpha * Normalized Betweenness Centrality.
        """
        self.alpha = alpha

    def compute_betweenness_centrality(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Constructs a NetworkX graph from the adjacency matrix and computes
        betweenness centrality for all nodes, normalized to [0, 1].
        """
        N = adj_matrix.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        # Build edges
        for i in range(N):
            for j in range(i + 1, N):
                if adj_matrix[i, j] > 0:
                    G.add_edge(i, j)

        # Compute betweenness centrality using NetworkX
        centrality_dict = nx.betweenness_centrality(G, normalized=True)
        centrality = np.array([centrality_dict[i] for i in range(N)], dtype=np.float32)
        
        # Normalize to [0, 1] relative to the max centrality in the graph
        max_c = float(np.max(centrality))
        if max_c > 0:
            centrality = centrality / max_c
            
        return centrality

    def retrieve_context(
        self,
        query_vector: np.ndarray,
        node_embeddings: np.ndarray,
        adj_matrix: np.ndarray,
        papers: List[Dict[str, Any]],
        top_k: int = 5,
        max_neighbors_to_include: int = 3,
        query: str = ""
    ) -> Dict[str, Any]:
        """
        Performs topology-aware context retrieval.
        
        Algorithm:
          1. Calculates cosine similarity of the query vector with GCN node embeddings.
          2. Selects top-K semantically closest nodes.
          3. Expands search to the 1-hop neighborhood of these top-K nodes.
          4. For all neighbors, computes a hybrid score boosting nodes with high betweenness centrality.
          5. Selects up to `max_neighbors_to_include` high-centrality neighborhood bottleneck papers.
          6. Merges and formats the results into a markdown context payload.
        
        Args:
            query_vector: Dense vector representing the query, shape (out_dim,).
            node_embeddings: Dense GCN embeddings matrix, shape (N, out_dim).
            adj_matrix: Adjacency matrix of the graph, shape (N, N).
            papers: List of metadata dictionaries for each paper in the graph.
            top_k: Number of initial semantic hits to retrieve.
            max_neighbors_to_include: Max number of bottleneck neighbors to explicitly inject.
            
        Returns:
            Dict containing retrieved nodes metadata, scores, and formatted Markdown prompt payload.
        """
        N = len(papers)
        if N == 0:
            return {"retrieved_nodes": [], "markdown_payload": "No context available (empty graph)."}

        # 1. Calculate cosine similarity in GCN embedding space
        q_norm = np.linalg.norm(query_vector)
        q_norm = 1e-9 if q_norm == 0 else q_norm
        emb_norms = np.linalg.norm(node_embeddings, axis=1)
        emb_norms = np.where(emb_norms == 0, 1e-9, emb_norms)
        
        similarities = np.dot(node_embeddings, query_vector) / (emb_norms * q_norm)
        similarities = np.clip(similarities, -1.0, 1.0)

        # 2. Compute betweenness centrality via NetworkX
        centrality = self.compute_betweenness_centrality(adj_matrix)

        # 3. Find top-K semantic hits
        semantic_ranks = np.argsort(similarities)[::-1]
        top_k_indices = list(semantic_ranks[:top_k])

        # 4. Construct NetworkX graph to traverse neighborhood
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if adj_matrix[i, j] > 0:
                    G.add_edge(i, j)

        # Traverse 1-hop neighbors of our semantic hits
        neighborhood_indices: Set[int] = set()
        for idx in top_k_indices:
            neighborhood_indices.update(G.neighbors(idx))
            
        # Exclude nodes that are already in our top-K semantic hits
        candidate_neighbors = neighborhood_indices.difference(set(top_k_indices))

        # 5. Score neighbors based on hybrid score (relevance + topology boost)
        neighbor_scores: List[Tuple[int, float, float, float]] = []
        for idx in candidate_neighbors:
            sim = float(similarities[idx])
            bc = float(centrality[idx])
            hybrid = sim + self.alpha * bc
            neighbor_scores.append((idx, hybrid, sim, bc))

        # Sort neighbors by hybrid score descending and pick the top bottleneck/relevant nodes
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        selected_neighbor_indices = [item[0] for item in neighbor_scores[:max_neighbors_to_include]]

        # 6. Merge initial hits and bottleneck neighbors
        final_retrieved_nodes: List[Dict[str, Any]] = []
        
        # Helper to format nodes
        def format_node(idx: int, role: str) -> Dict[str, Any]:
            p = papers[idx]
            sim = float(similarities[idx])
            bc = float(centrality[idx])
            
            node_data = {
                "idx": int(idx),
                "id": p["id"],
                "title": p["title"],
                "authors": p["authors"],
                "summary": p["summary"],
                "categories": p["categories"],
                "semantic_similarity": sim,
                "betweenness_centrality": bc,
                "hybrid_score": sim + self.alpha * bc,
                "role": role
            }
            return node_data

        for idx in top_k_indices:
            final_retrieved_nodes.append(format_node(idx, "Semantic Hit"))
            
        for idx in selected_neighbor_indices:
            final_retrieved_nodes.append(format_node(idx, "Structural Bridge (Bottleneck)"))

        # Sort the final retrieved nodes by hybrid score to feed the LLM in order of combined relevance
        final_retrieved_nodes.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return {
            "retrieved_nodes": final_retrieved_nodes
        }

    def format_payload(self, query: str, retrieved_nodes: List[Dict[str, Any]], original_papers: List[Dict[str, Any]], instruction: str = "") -> Dict[str, Any]:
        """
        Formats the retrieved nodes into a markdown payload and JSON data structure.
        Performs micro-retrieval on the LaTeX trees if they are present in the original_papers.
        """
        final_nodes = []
        semantic_count = sum(1 for n in retrieved_nodes if n["role"] == "Semantic Hit")
        bridge_count = sum(1 for n in retrieved_nodes if "Bridge" in n["role"])
        
        for node in retrieved_nodes:
            idx = node["idx"]
            p = original_papers[idx]
            
            # Create a copy so we don't mutate the retrieval output
            node_data = node.copy()
            
            # Stage 2 (Micro): Granular Tree Retrieval
            if "tree" in p:
                node_data["relevant_branches"] = self._micro_retrieve_tree(query, p["tree"], instruction=instruction)
            
            final_nodes.append(node_data)
            
        has_trees = any("relevant_branches" in n for n in final_nodes)
        
        if has_trees:
            # Format as JSON payload for LLM consumption
            payload_data = {
                "metrics": {
                    "semantic_hits": semantic_count,
                    "structural_bridges": bridge_count
                },
                "nodes": final_nodes
            }
            payload_markdown = json.dumps(payload_data, indent=2)
        else:
            # 7. Generate structured Markdown Prompt Context Payload
            payload_lines = [
                "### GRAPH-RAG CONTEXT PAYLOAD",
                "The following relevant literature and structural bridge papers have been retrieved from the citation network.",
                f"**Retrieval Metrics**: Semantic Hits: {semantic_count} | Structural Bridges Injected: {bridge_count}\n"
            ]

            for rank, node in enumerate(final_nodes, 1):
                role_badge = f"[{node['role'].upper()}]"
                authors_str = ", ".join(node['authors'])
                categories_str = ", ".join(node['categories'])
                
                payload_lines.append(
                    f"#### [{rank}] {node['title']}\n"
                    f"- **arXiv ID**: {node['id']} | **Role**: {role_badge}\n"
                    f"- **Authors**: {authors_str}\n"
                    f"- **Categories**: {categories_str}\n"
                    f"- **Metrics**: Semantic Similarity: {node['semantic_similarity']:.4f} | "
                    f"Normalized Betweenness Centrality: {node['betweenness_centrality']:.4f} | "
                    f"Hybrid Score: {node['hybrid_score']:.4f}\n"
                    f"- **Abstract Summary**:\n"
                    f"  > {node['summary']}\n"
                )

            payload_markdown = "\n".join(payload_lines)

        return {
            "retrieved_nodes": final_nodes,
            "markdown_payload": payload_markdown,
            "payload_data": payload_data if has_trees else None
        }

    def _cosine_similarity_text(self, text1: str, text2: str) -> float:
        """Fallback lexical cosine similarity if SentenceTransformers is not available."""
        vec1 = Counter(text1.lower().split())
        vec2 = Counter(text2.lower().split())
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return float(numerator) / denominator

    def _micro_retrieve_tree(self, query: str, tree: Dict[str, Any], top_branches: int = 2, instruction: str = "") -> List[Dict[str, Any]]:
        """
        Stage 2 (Micro): Traverses the internal tree and extracts the most relevant branches based on semantic similarity.
        """
        branches = tree.get("branches", [])
        if not branches:
            return []
            
        scored_branches = []
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, 'semantic_model'):
                self.semantic_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
                
            branch_texts = [b.get("title", "") + " " + " ".join(item.get("text", "") for item in b.get("content", [])) for b in branches]
            
            # Batch encode with instruction formatting for query
            formatted_query = f"{instruction} {query}".strip()
            embeddings = self.semantic_model.encode([formatted_query] + branch_texts, convert_to_numpy=True)
            q_emb = embeddings[0]
            branch_embs = embeddings[1:]
            
            # Normalize and dot product
            q_norm = q_emb / np.linalg.norm(q_emb)
            branch_norms = branch_embs / np.linalg.norm(branch_embs, axis=1, keepdims=True)
            similarities = np.dot(branch_norms, q_norm)
            
            for i, branch in enumerate(branches):
                scored_branches.append((float(similarities[i]), branch))
        except ImportError:
            # Fallback to lexical
            for branch in branches:
                branch_text = branch.get("title", "") + " " + " ".join(item.get("text", "") for item in branch.get("content", []))
                sim = self._cosine_similarity_text(query, branch_text)
                scored_branches.append((sim, branch))

        scored_branches.sort(key=lambda x: x[0], reverse=True)
        return [b[1] for b in scored_branches[:top_branches]]

    def visualize_query_subgraph(
        self,
        adj_matrix: np.ndarray,
        retrieved_nodes: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
        output_path: str = "query_graph.png"
    ) -> None:
        """
        Generates a matplotlib network visualization of the citation graph,
        highlighting semantic hits and structural bottleneck nodes.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        N = adj_matrix.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(i + 1, N):
                if adj_matrix[i, j] > 0:
                    G.add_edge(i, j)

        # Create mapping of node index to role
        node_roles = {i: "Other" for i in range(N)}
        node_labels = {}
        for rank, node in enumerate(retrieved_nodes, 1):
            idx = node["idx"]
            node_roles[idx] = node["role"]
            # Label with rank and short arXiv ID
            node_labels[idx] = f"[{rank}] {node['id']}"

        # Assign colors based on roles
        node_colors = []
        node_sizes = []
        for i in range(N):
            role = node_roles[i]
            if role == "Semantic Hit":
                node_colors.append("#2B6CB0")  # Royal Blue
                node_sizes.append(400)
            elif role == "Structural Bridge (Bottleneck)":
                node_colors.append("#D69E2E")  # Gold / Amber
                node_sizes.append(500)
            else:
                node_colors.append("#E2E8F0")  # Light Gray
                node_sizes.append(150)

        plt.figure(figsize=(10, 8))
        
        # Calculate layout
        pos = nx.spring_layout(G, seed=42, k=1.5/np.sqrt(N))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#CBD5E0")
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors, 
            node_size=node_sizes, 
            edgecolors="#718096", 
            linewidths=0.5
        )
        
        # Draw labels only for retrieved/important nodes to avoid clutter
        nx.draw_networkx_labels(
            G, pos, 
            labels=node_labels, 
            font_size=8, 
            font_weight="bold", 
            font_family="sans-serif"
        )
        
        # Create legend handles manually
        blue_patch = mpatches.Patch(color='#2B6CB0', label='Semantic Hit')
        gold_patch = mpatches.Patch(color='#D69E2E', label='Structural Bridge (Bottleneck)')
        gray_patch = mpatches.Patch(color='#E2E8F0', label='Other Papers')
        plt.legend(handles=[blue_patch, gold_patch, gray_patch], loc='upper right')
        
        plt.title("Graph-RAG Citation Network & Topology-Aware Query Retrieval", fontsize=12, fontweight="bold", pad=15)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Citation graph visualization saved to: {output_path}")

