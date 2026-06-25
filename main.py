#!/usr/bin/env python3
"""
Main entry point for the Citation Cartography and Graph-RAG system.
Integrates Data Ingestion, GCN representation learning, and Topology-Aware retrieval.
"""
import argparse
import sys
import numpy as np
from typing import Dict, Any, List, Tuple

# Import modular components
from src.ingestion import ArXivIngestionPipeline, generate_mock_data, generate_hierarchical_mock_data
from src.gcn import train_gcn_unsupervised
from src.gat import train_gat_unsupervised
from src.retrieval import GraphRAGRetrievalEngine
from src.formatter import ContextFormatter
from src.synthesis import SemanticSynthesizer

def generate_arxiv_query(search_text: str) -> str:
    """
    Automatically generates a robust arXiv API query from a natural language search string.
    """
    stop_words = {"how", "do", "what", "why", "is", "are", "for", "in", "the", "a", "an", "of", "and", "to", "with", "on", "by", "using", "about", "can", "you", "find"}
    words = search_text.replace("'", "").replace('"', '').replace('?', '').split()
    keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    
    if not keywords:
        return 'all:"computer science"' # safe fallback
        
    # Join keywords with AND
    query_parts = [f"all:{kw}" for kw in keywords]
    return " AND ".join(query_parts)

def project_query_to_model_space(q_feat: np.ndarray, params: Any, args: argparse.Namespace) -> np.ndarray:
    """
    Inductively projects a query feature vector into the trained embedding space.
    """
    if args.model == "gcn":
        w0 = np.array(params['GCNLayer_0']['weights'])
        b0 = np.array(params['GCNLayer_0']['bias'])
        w1 = np.array(params['GCNLayer_1']['weights'])
        b1 = np.array(params['GCNLayer_1']['bias'])
        
        # Layer 1 dense projection + ReLU
        h1 = np.dot(q_feat, w0) + b0
        h1 = np.maximum(h1, 0.0)
        
        # Layer 2 dense projection
        h2 = np.dot(h1, w1) + b1
        return h2
    elif args.model == "gat":
        from src.gat import GAT, dense_to_jraph
        
        # Create an isolated graph containing just the query
        q_graph = dense_to_jraph(q_feat.reshape(1, -1), np.zeros((1, 1)))
        
        model = GAT(
            hidden_dim=args.hidden_dim, 
            out_dim=args.out_dim, 
            num_heads=args.heads, 
            dropout_rate=args.dropout
        )
        
        # Run deterministic forward pass to get the embedding
        out_graph = model.apply({'params': params}, q_graph, deterministic=True)
        return np.array(out_graph.nodes).flatten()
    else:
        raise ValueError(f"Unknown model type: {args.model}")

def run_pipeline(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("      CITATION CARTOGRAPHY & TOPOLOGY-AWARE GRAPH-RAG PIPELINE      ")
    print("=" * 70)

    # 1. Ingestion Phase
    papers: List[Dict[str, Any]] = []
    A: np.ndarray = np.empty((0, 0))
    X: np.ndarray = np.empty((0, 0))
    pipeline = ArXivIngestionPipeline(delay=3.0)

    if args.mock:
        print(f"[Ingestion] Generating {args.limit} nodes of synthetic mock data...")
        papers, A, X = generate_hierarchical_mock_data(max_features=args.features, pipeline=pipeline)
    else:
        print(f"[Ingestion] Fetching up to {args.limit} papers from arXiv API for query: '{args.query}'...")
        try:
            # Fetch from arXiv API
            papers = pipeline.fetch_metadata(query=args.query, limit=args.limit)
            if not papers:
                raise ValueError("No papers returned from arXiv API.")
                
            print(f"[Ingestion] Successfully fetched {len(papers)} papers. Constructing graph structures...")
            # Build graph A and feature matrix X
            A, X = pipeline.build_graph(papers, max_features=args.features, sim_threshold=0.20)
            print(f"[Ingestion] Graph construction complete. Adjacency: {A.shape} | Features: {X.shape}")
        except Exception as e:
            print(f"\n[Warning] Live arXiv API ingestion failed or timed out: {e}")
            print(f"[Warning] Falling back to synthetic mock dataset automatically.\n")
            papers, A, X = generate_hierarchical_mock_data(max_features=args.features, pipeline=pipeline)

    N = len(papers)
    if N == 0:
        print("[Error] No papers loaded. Exiting.")
        sys.exit(1)

    print(f"\n[Graph Analysis] Nodes (Papers): {N}")
    print(f"[Graph Analysis] Edges: {int(np.sum(A) / 2)} (Symmetric)")
    print(f"[Graph Analysis] Density: {np.sum(A) / (N * (N - 1)):.4f}")

    # 2. Representation Learning Phase
    if args.model == "gat":
        print(f"\n[GAT Training] Initializing 2-layer GAT Model (Hidden: {args.hidden_dim}, Out: {args.out_dim})...")
        print(f"[GAT Training] Training unsupervised node representations via Link Prediction (Epochs: {args.epochs})...")
        
        embeddings, params, metrics = train_gat_unsupervised(
            X=X,
            A=A,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            num_heads=args.heads,
            dropout_rate=args.dropout
        )
        
        final_loss = metrics["loss"][-1] if metrics["loss"] else 0.0
        print(f"[GAT Training] Complete. Final Link Reconstruction Loss: {final_loss:.4f}")
        print(f"[GAT Training] Resulting Node Embeddings Matrix shape: {embeddings.shape}")
    else:
        print(f"\n[GCN Training] Initializing 2-layer GCN Model (Hidden: {args.hidden_dim}, Out: {args.out_dim})...")
        print(f"[GCN Training] Training unsupervised node representations via Link Prediction (Epochs: {args.epochs})...")
        
        embeddings, params, metrics = train_gcn_unsupervised(
            X=X,
            A=A,
            hidden_dim=args.hidden_dim,
            out_dim=args.out_dim,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed
        )
        
        final_loss = metrics["loss"][-1] if metrics["loss"] else 0.0
        print(f"[GCN Training] Complete. Final Link Reconstruction Loss: {final_loss:.4f}")
        print(f"[GCN Training] Resulting Node Embeddings Matrix shape: {embeddings.shape}")

    # 3. Context Retrieval Engine Phase
    print(f"\n[Retrieval] Vectorizing and projecting search query: '{args.search}'...")
    
    # Vectorize text query into raw bag-of-words/TF-IDF feature vector
    q_feat = pipeline.vectorize_query(args.search)
    
    # Project raw query features into embedding space
    q_emb = project_query_to_model_space(q_feat, params, args)

    print(f"[Retrieval] Query representation constructed. Features norm: {np.linalg.norm(q_feat):.4f} | Embedding norm: {np.linalg.norm(q_emb):.4f}")
    print(f"[Retrieval] Executing topology-aware search (Top-K: {args.top_k}, Injected Bridges: {args.bridges}, Alpha: {args.alpha})...")

    # Initialize retrieval engine
    retriever = GraphRAGRetrievalEngine(alpha=args.alpha)
    retrieval_results = retriever.retrieve_context(
        query_vector=q_emb,
        node_embeddings=embeddings,
        adj_matrix=A,
        papers=papers,
        top_k=args.top_k,
        max_neighbors_to_include=args.bridges,
        query=args.search
    )

    # 3.1 Generate graph visualization
    retriever.visualize_query_subgraph(
        adj_matrix=A,
        retrieved_nodes=retrieval_results["retrieved_nodes"],
        papers=papers,
        output_path=args.viz_output
    )

    # 4. Display Results
    print("\n" + "=" * 70)
    print("                     RETRIEVED GRAPH-RAG CONTEXT                    ")
    print("=" * 70)
    print("=" * 70)
    
    # 5. Semantic Synthesis Phase
    print(f"\n[Synthesis] Flattening RAG context for LLM ingestion...")
    payload = retrieval_results.get("payload_data", {})
    if payload:
        formatted_context = ContextFormatter.format_payload(payload)
    else:
        formatted_context = "No structural hierarchical trees extracted. Please review flat citations instead."
    
    print(f"[Synthesis] Passing context to local Ollama Engine (gemma4:31b-cloud) in '{args.mode}' mode...")
    synthesizer = SemanticSynthesizer()
    synthesizer.generate_synthesis(query=args.search, formatted_context=formatted_context, mode=args.mode)

    print("\n[Success] End-to-end Graph-RAG pipeline completed successfully.")
    print("=" * 70)

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Research Graph-RAG Pipeline.")
    parser.add_argument(
        "--query", 
        type=str, 
        default=None,
        help="Seed query for arXiv API search."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gcn",
        choices=["gcn", "gat"],
        help="Graph neural network model to use for embeddings (gcn or gat)."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=50,
        help="Max number of papers to crawl/generate."
    )
    parser.add_argument(
        "--features", 
        type=int, 
        default=128,
        help="Dimensionality of raw input TF-IDF features."
    )
    parser.add_argument(
        "--mock", 
        action="store_true",
        help="Force using offline mock data generator."
    )
    parser.add_argument(
        "--hidden-dim", 
        type=int, 
        default=64,
        help="GCN first layer hidden dimension."
    )
    parser.add_argument(
        "--out-dim", 
        type=int, 
        default=32,
        help="GCN second layer output/embedding dimension."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=150,
        help="Number of epochs for GCN unsupervised training."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01,
        help="Adam optimizer learning rate."
    )
    parser.add_argument(
        "--heads", 
        type=int, 
        default=4,
        help="Number of attention heads (GAT only)."
    )
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.6,
        help="Dropout rate (GAT only)."
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.4,
        help="Weighting factor for topology centrality boost."
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=5,
        help="Number of primary semantic hits to retrieve."
    )
    parser.add_argument(
        "--bridges", 
        type=int, 
        default=3,
        help="Maximum structural bottleneck neighbors to inject."
    )
    parser.add_argument(
        "--search", 
        type=str, 
        default=None,
        help="RAG query text to retrieve context for."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="PRNG seed for GCN weights initialization."
    )
    parser.add_argument(
        "--viz-output", 
        type=str, 
        default="query_graph.png",
        help="Output filepath for the query graph network visualization."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="review",
        choices=["review", "curriculum"],
        help="Output mode: 'review' for literature review, 'curriculum' for structured reading syllabus."
    )
    
    args = parser.parse_args()
    
    # Interactive Prompts
    if args.search is None:
        args.search = input("\n[Research Topic] What would you like to research? (e.g. 'How do graph attention networks work?'): ").strip()
        if not args.search:
            print("Error: A search query is required.")
            return

    # Auto-generate arXiv seed query if not provided and not using mock data
    if args.query is None and not args.mock:
        args.query = generate_arxiv_query(args.search)
        print(f"[Auto-Detect] Generated arXiv ingestion query: {args.query}")
        
    run_pipeline(args)

if __name__ == "__main__":
    main()
