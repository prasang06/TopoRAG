# Hierarchical Graph-RAG Pipeline for Scientific Literature

A topology-aware Retrieval-Augmented Generation (RAG) architecture tailored for deep scientific literature research. This pipeline ingests raw LaTeX source code from arXiv, constructs a citation graph, learns structural embeddings via unsupervised Graph Neural Networks (GCN/GAT), utilizes instruction-tuned dense embeddings (BGE-Large) for asymmetric search, and synthesizes fully-cited literature reviews or structured curricula through a modern Next.js Web Interface.

## Architecture Overview

The pipeline operates in six sequential phases:

1. **Ingestion & Parsing**: Asynchronously fetches e-print tarballs from the arXiv API with polite rate-limiting and persistent local `.cache` storage. A custom regex-based parser extracts the hierarchical anatomy of the paper (Sections, Subsections, Paragraphs, and Equations).
2. **Graph Construction**: Builds a directed `NetworkX` citation graph representing the structural dependencies between papers.
3. **Graph Representation Learning**: Trains an unsupervised Graph Neural Network (GCN/GAT) using `JAX` and `Jraph`. The model learns node embeddings via a Link Prediction objective.
4. **Instruction-Tuned Dense Embedding**: Uses `BAAI/bge-large-en-v1.5` (running on CPU to prevent CUDA OOM) to perform an asymmetric semantic search by prepending specific task instructions to the user's query vector.
5. **Topology-Aware Retrieval**: 
    - *Macro-Retrieval*: Combines dense semantic similarity with network betweenness-centrality to retrieve highly relevant "Semantic Hits" alongside foundational "Structural Bottlenecks".
    - *Micro-Retrieval*: Traverses the internal LaTeX Tree of the selected papers to extract specific branches or equations matching the query.
6. **Semantic Synthesis**: Formats the extracted hierarchical context and streams it to a local LLM via Ollama.

## Prerequisites

- Python 3.10+
- Node.js & npm (for the frontend interface)
- [Ollama](https://ollama.com/) (running locally, default model: `gemma4:31b-cloud`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/prasang06/Citation-Cartography-and-RAG-Graphing.git
cd Citation-Cartography-and-RAG-Graphing
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: The `bge-large` embedding model will be downloaded to your local huggingface cache automatically upon first run, which is ~1.3GB).*

## Usage

Start the fullstack application automatically with the unified startup script. Ensure the Ollama daemon is running (`ollama serve`) in a separate terminal.

```bash
chmod +x start.sh
./start.sh
```

This script will automatically:
1. Verify Ollama is active.
2. Spin up the FastAPI backend (`uvicorn`) on port `8000`.
3. Install missing Node dependencies and spin up the Next.js frontend on port `3000`.
4. Open the interactive web interface in your default browser.

Press `Ctrl+C` in the terminal to gracefully shutdown both frontend and backend processes.

## CLI Usage (Headless Mode)

If you prefer to run the raw backend engine without the graphical interface:

```bash
python main.py --search "quantum error correction surface codes" --mode curriculum --limit 10
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--search` | Free-text research idea or query | *Prompted* |
| `--mode` | Output mode (`review` or `curriculum`) | `review` |
| `--limit` | Max number of papers to crawl/generate | `50` |
| `--model` | Graph neural network model (`gcn` or `gat`) | `gcn` |
| `--mock` | Skip arXiv and use synthetic data | `False` |

## Output
- **Web Interface**: Real-time generation of the syllabus and dynamic interactive Graph Visualization of the paper network.
