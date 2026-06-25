# Hierarchical Graph-RAG Pipeline for Scientific Literature

A topology-aware Retrieval-Augmented Generation (RAG) architecture tailored for deep scientific literature research. This pipeline ingests raw LaTeX source code from arXiv, constructs a citation graph, learns structural embeddings via unsupervised Graph Neural Networks (GCN/GAT), and synthesizes fully-cited literature reviews or structured reading curricula using local LLMs.

## Architecture Overview

The pipeline operates in five sequential phases:

1. **Ingestion & Parsing**: Asynchronously fetches e-print tarballs from the arXiv API. A custom regex-based parser extracts the hierarchical anatomy of the paper (Sections, Subsections, Paragraphs, and Equations) into a nested Tree structure.
2. **Graph Construction**: Builds a directed `NetworkX` citation graph representing the structural dependencies between papers.
3. **Graph Representation Learning**: Trains an unsupervised Graph Neural Network (GCN by default, or GAT for noisy/dense graphs) using `JAX` and `Jraph`. The model learns node embeddings via a Link Prediction objective, capturing the topology of the citation network.
4. **Topology-Aware Retrieval**: 
    - *Macro-Retrieval*: Combines TF-IDF semantic similarity with network betweenness-centrality to retrieve highly relevant "Semantic Hits" alongside foundational "Structural Bottlenecks".
    - *Micro-Retrieval*: Traverses the internal LaTeX Tree of the selected papers to extract specific branches or equations matching the query.
5. **Semantic Synthesis**: Formats the extracted hierarchical context and streams it to a local LLM via Ollama to generate rigorously cited output.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) (running locally)
- Pulled local model (Defaults to `gemma4:31b-cloud`, but can be modified in `src/synthesis.py`).

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

*(Note: The default `jaxlib` is CPU-only. If you have an NVIDIA GPU and wish to accelerate the GNN training, install the CUDA runtime: `pip install -U "jax[cuda12]"`).*

## Usage

Start the pipeline via the CLI. Ensure the Ollama daemon is running (`ollama serve`).

### 1. Literature Review Mode (Default)
Generates a cohesive, narrative literature review synthesizing the foundational theories and specific applications.

```bash
python main.py --search "quantum computing error correction surface codes" --mode review --limit 10
```

### 2. Curriculum Generation Mode
Generates a structured, sequential reading syllabus explicitly tailored to a proposed research project.

```bash
python main.py --search "I want to build a simulator for surface codes" --mode curriculum --limit 5
```

### 3. Mock Data Testing
To run the pipeline without hitting the arXiv API, use the `--mock` flag to generate a synthetic citation graph:

```bash
python main.py --mock --mode curriculum --search "quantum computing equation"
```

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--search` | Free-text research idea or query | *Prompted* |
| `--mode` | Output mode (`review` or `curriculum`) | `review` |
| `--limit` | Max number of papers to crawl/generate | `50` |
| `--model` | Graph neural network model (`gcn` or `gat`) | `gcn` |
| `--mock` | Skip arXiv and use synthetic data | `False` |
| `--top-k` | Number of primary semantic hits to retrieve | `5` |
| `--bridges`| Max structural bottleneck neighbors to inject | `3` |
| `--alpha` | Weighting factor for topology centrality boost | `0.4` |

## Output
- **Terminal**: Real-time streaming of the synthesis engine.
- **Visual**: Saves a NetworkX visualization of the active query subgraph to `query_graph.png`.
