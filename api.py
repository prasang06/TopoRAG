import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import numpy as np
import argparse

from src.ingestion import ArXivIngestionPipeline, generate_hierarchical_mock_data
from src.gat import train_gat_unsupervised
from src.retrieval import GraphRAGRetrievalEngine
from src.formatter import ContextFormatter
from src.synthesis import SemanticSynthesizer

app = FastAPI(title="Graph-RAG API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    mode: str = "review"
    limit: int = 5

class SynthesisRequest(BaseModel):
    query: str
    formatted_context: str
    mode: str = "review"

def generate_arxiv_query(search_text: str) -> str:
    # 1. Clean query of punctuation that might break arXiv API
    import re
    clean_text = re.sub(r'[^\w\s]', '', search_text)
    
    # 2. Extract most meaningful keywords
    words = clean_text.split()
    stop_words = {"i", "want", "to", "work", "on", "open", "problem", "given", "copies", "of", "a", "an", "the", "and", "or", "whose", "that", "this", "give", "me", "resources"}
    
    # Filter and sort by length (heuristically, longer words are more specific jargon)
    keywords = [w for w in words if w.lower() not in stop_words and len(w) > 4]
    keywords = sorted(keywords, key=len, reverse=True)[:2]
    
    # Fallback if no keywords
    if not keywords:
        return 'all:"quantum"'
        
    query_parts = [f"all:{kw}" for kw in keywords]
    return " AND ".join(query_parts)

@app.post("/api/search")
def search(request: SearchRequest):
    try:
        arxiv_query = generate_arxiv_query(request.query)
        pipeline = ArXivIngestionPipeline(delay=1.0) # Faster delay for API
        
        # 1. Ingestion (Shallow Fetch)
        try:
            print(f"Fetching papers for query: {arxiv_query}")
            papers = pipeline.fetch_metadata(query=arxiv_query, limit=request.limit, fetch_multiplier=4)
            if not papers:
                raise ValueError("No papers found")
            A, X = pipeline.build_graph(papers, max_features=128, sim_threshold=0.20)
        except Exception as e:
            print(f"arXiv API failed: {e}. Falling back to mock data.")
            papers, A, X = generate_hierarchical_mock_data(max_features=128, pipeline=pipeline)

        # 2. GAT Training (Metadata-Only)
        embeddings, params, metrics = train_gat_unsupervised(
            X=X, A=A, hidden_dim=64, out_dim=32, epochs=100, lr=0.01, seed=42, num_heads=4, dropout_rate=0.6
        )

        # 3. Project Query
        from src.gat import GAT, dense_to_jraph
        q_feat = pipeline.vectorize_query(request.query)
        q_graph = dense_to_jraph(q_feat.reshape(1, -1), np.zeros((1, 1)))
        model = GAT(hidden_dim=64, out_dim=32, num_heads=4, dropout_rate=0.6)
        out_graph = model.apply({'params': params}, q_graph, deterministic=True)
        q_emb = np.array(out_graph.nodes).flatten()

        # 4. Retrieval (Select Top Nodes)
        retriever = GraphRAGRetrievalEngine(alpha=0.4)
        dynamic_top_k = min(15, max(5, int(request.limit * 0.6)))
        max_neighbors = max(1, request.limit - dynamic_top_k)
        
        retrieval_results = retriever.retrieve_context(
            query_vector=q_emb,
            node_embeddings=embeddings,
            adj_matrix=A,
            papers=papers,
            top_k=dynamic_top_k,
            max_neighbors_to_include=max_neighbors,
            query=request.query
        )
        
        top_nodes = retrieval_results["retrieved_nodes"]
        retrieved_indices = [node["idx"] for node in top_nodes]
        
        # 5. Deep Fetch (LaTeX parsing only for winners)
        top_papers = [papers[i] for i in retrieved_indices]
        pipeline.fetch_latex_for_papers(top_papers)
        
        # 6. Formatting Payload
        formatted_results = retriever.format_payload(request.query, top_nodes, papers)

        # Format Graph for Frontend
        subgraph_nodes = []
        for node_dict in formatted_results["retrieved_nodes"]:
            real_idx = node_dict["idx"]
            p = papers[real_idx]
            role = node_dict.get("role", "Semantic Hit")
            
            node_payload = {
                "id": p["id"],
                "title": p.get("title", f"Paper {p['id']}"),
                "summary": p.get("summary", ""),
                "authors": p.get("authors", []),
                "role": role,
                "val": 2 if "Bottleneck" in role or "Bridge" in role else 1
            }
            if "relevant_branches" in node_dict:
                node_payload["relevant_branches"] = node_dict["relevant_branches"]
            
            subgraph_nodes.append(node_payload)

        subgraph_links = []
        for i in retrieved_indices:
            for j in retrieved_indices:
                if A[i, j] > 0 and i != j:
                    subgraph_links.append({
                        "source": papers[i]["id"],
                        "target": papers[j]["id"],
                        "weight": float(A[i, j])
                    })

        graph_data = {"nodes": subgraph_nodes, "links": subgraph_links}
        
        # Prepare context payload
        formatted_context = formatted_results["markdown_payload"]

        return {
            "graph": graph_data,
            "trees": {},
            "formatted_context": formatted_context
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthesize")
async def synthesize(request: SynthesisRequest):
    synthesizer = SemanticSynthesizer()
    
    # We will use an asynchronous generator to yield chunks
    async def event_generator():
        import ollama
        from src.synthesis import SYSTEM_PROMPT, CURRICULUM_PROMPT
        
        try:
            if request.mode == "curriculum":
                system_prompt = CURRICULUM_PROMPT
                task_desc = "reading curriculum"
            else:
                system_prompt = SYSTEM_PROMPT
                task_desc = "literature review"

            user_prompt = f"USER QUERY / PROJECT IDEA: {request.query}\n\n=== EXTRACTED CONTEXT ===\n{request.formatted_context}\n\nPlease generate the {task_desc} now."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Since the ollama client in `src/synthesis.py` uses `stream=True`, we can just iterate over it.
            response = ollama.chat(
                model=synthesizer.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.2,
                    "num_predict": 1024,
                }
            )
            for chunk in response:
                text = chunk['message']['content']
                yield text
                await asyncio.sleep(0.01) # Yield control to event loop
                
        except Exception as e:
            yield f"\n[Error] LLM Synthesis failed: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
