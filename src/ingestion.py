import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import re
import math
from collections import Counter
from typing import Dict, Any, List, Tuple, Set
import numpy as np
import asyncio
from src.latex_parser import fetch_arxiv_eprint, parse_latex_to_tree, generate_fallback_tree


class ArXivIngestionPipeline:
    """
    Handles ingestion of metadata from the arXiv API, parses the response,
    and constructs the citation/relationship graph and TF-IDF feature matrices.
    """
    def __init__(self, base_url: str = "http://export.arxiv.org/api/query", delay: float = 3.0):
        self.base_url = base_url
        self.delay = delay
        self.last_request_time = 0.0

    def _polite_request(self, url: str) -> str:
        """f
        Executes an HTTP GET request with a polite delay (rate limiting).
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            time.sleep(sleep_time)
        
        try:
            self.last_request_time = time.time()
            headers = {"User-Agent": "CitationCartographyAgent/1.0 (prasang@example.com)"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.read().decode('utf-8')
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to arXiv API: {e}") from e

    def fetch_metadata(self, query: str, limit: int = 50, page_size: int = 25, fetch_multiplier: int = 4) -> List[Dict[str, Any]]:
        """
        Fetches arXiv papers matching a seed query using pagination and rate limiting.
        Fetches `limit * fetch_multiplier` papers to build a large metadata pool for GAT ranking.
        Does NOT download the heavy LaTeX trees.
        """
        all_papers: List[Dict[str, Any]] = []
        start = 0
        target_count = limit * fetch_multiplier

        while len(all_papers) < target_count:
            batch_limit = min(page_size, target_count - len(all_papers))
            # Format query parameters
            # arXiv uses standard URL encoding
            encoded_query = urllib.parse.quote(query)
            url = f"{self.base_url}?search_query={encoded_query}&start={start}&max_results={batch_limit}"
            
            try:
                xml_data = self._polite_request(url)
                batch_papers = self._parse_arxiv_xml(xml_data)
                if not batch_papers:
                    break
                all_papers.extend(batch_papers)
                start += len(batch_papers)
                if len(batch_papers) < batch_limit:
                    break  # No more results available
            except Exception as e:
                if all_papers:
                    print(f"Warning: Fetch interrupted ({e}). Using {len(all_papers)} papers fetched so far.")
                    break
                else:
                    raise e
                    
        papers = all_papers[:target_count]
        print(f"[Ingestion] Fetched {len(papers)} metadata records for topological ranking.")
        return papers

    def fetch_latex_for_papers(self, papers: List[Dict[str, Any]]):
        """
        Asynchronously fetches and parses the full LaTeX e-prints for a specific subset of papers.
        This modifies the passed dictionary in-place by adding the 'tree' key.
        """
        print(f"[Ingestion] Deep fetching and parsing full LaTeX trees for {len(papers)} top-ranked papers...")
        asyncio.run(self._fetch_all_trees(papers))

    async def _fetch_all_trees(self, papers: List[Dict[str, Any]]):
        async def process_paper(paper):
            arxiv_id = paper['id']
            tex_content = await fetch_arxiv_eprint(arxiv_id)
            if tex_content:
                paper['tree'] = parse_latex_to_tree(tex_content, paper.copy())
            else:
                paper['tree'] = generate_fallback_tree(paper.copy())
                
        # Run in parallel with a small concurrency limit to prevent 429 errors from arXiv
        semaphore = asyncio.Semaphore(2)
        async def sem_process(p):
            async with semaphore:
                await process_paper(p)
                await asyncio.sleep(1.0) # Polite delay between requests
                
        await asyncio.gather(*(sem_process(p) for p in papers))

    def _parse_arxiv_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parses XML metadata from the Atom feed returned by arXiv.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML response from arXiv: {e}") from e

        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers: List[Dict[str, Any]] = []

        for entry in root.findall('atom:entry', ns):
            # Extract raw ID and normalized short ID
            id_url_el = entry.find('atom:id', ns)
            id_str = id_url_el.text.strip() if id_url_el is not None and id_url_el.text else ""
            arxiv_id = id_str.split('/abs/')[-1].split('v')[0] if '/abs/' in id_str else id_str
            
            title_el = entry.find('atom:title', ns)
            title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else "Untitled"
            
            summary_el = entry.find('atom:summary', ns)
            summary = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else ""
            
            authors: List[str] = []
            for author in entry.findall('atom:author', ns):
                name_el = author.find('atom:name', ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())
                    
            categories: List[str] = []
            for cat in entry.findall('atom:category', ns):
                term = cat.attrib.get('term')
                if term:
                    categories.append(term)
                    
            papers.append({
                'id': arxiv_id,
                'title': title,
                'summary': summary,
                'authors': authors,
                'categories': categories,
                'raw_id': id_str
            })
        return papers

    def build_graph(
        self, 
        papers: List[Dict[str, Any]], 
        max_features: int = 1024,
        sim_threshold: float = 0.50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the adjacency matrix (A) and feature matrix (X) from parsed papers.
        
        Graph edges (A) are constructed using three layers of relation:
          1. Direct citations: paper A mentions paper B's arXiv ID in its abstract.
          2. Co-authorship: paper A and paper B share at least one author.
          3. Shared category + semantic textual similarity above `sim_threshold`.
        
        Node features (X) are dense semantic embeddings computed from title and summary.
        
        Returns:
            A: Adjacency matrix of shape (N, N), symmetric and unweighted (values 0 or 1).
            X: Feature matrix of shape (N, max_features) normalized to unit length.
        """
        N = len(papers)
        if N == 0:
            return np.empty((0, 0)), np.empty((0, max_features))

        # 1. Compute Node Features (X) using SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            # Load specialized scientific semantic embedding model
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            combined_texts = [f"{p['title']}[SEP]{p['summary']}" for p in papers]
            print(f"[Ingestion] Computing dense semantic embeddings for {len(combined_texts)} papers...")
            # Compute embeddings and return as numpy array
            X = model.encode(combined_texts, convert_to_numpy=True)
            # Normalize to unit length
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = np.divide(X, norms, out=np.zeros_like(X), where=norms!=0)
        except ImportError:
            print("[Warning] sentence-transformers not installed. Falling back to simple lexical TF-IDF.")
            combined_texts = [f"{p['title']}. {p['summary']}" for p in papers]
            X = self._build_tfidf_features(combined_texts, max_features)

        # 2. Build Adjacency Matrix (A)
        A = np.zeros((N, N))
        
        # Maps for fast lookup
        id_to_idx = {p['id']: idx for idx, p in enumerate(papers)}
        
        # Regex for modern and old arXiv ID patterns
        modern_pattern = re.compile(r'\b(?:arxiv:)?(\d{4}\.\d{4,5})\b', re.IGNORECASE)
        old_pattern = re.compile(r'\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})\b', re.IGNORECASE)

        for i, paper in enumerate(papers):
            # Direct Citation Links (from Summary/Abstract text)
            abstract = paper['summary']
            citations: Set[str] = set()
            for match in modern_pattern.finditer(abstract):
                citations.add(match.group(1))
            for match in old_pattern.finditer(abstract):
                citations.add(match.group(1))
                
            for cite_id in citations:
                if cite_id in id_to_idx:
                    j = id_to_idx[cite_id]
                    if i != j:
                        A[i, j] = 1.0
                        A[j, i] = 1.0

            # Co-authorship & Semantic Similarity Links
            for j in range(i + 1, N):
                other = papers[j]
                
                # Check co-authorship
                shared_authors = set(paper['authors']).intersection(set(other['authors']))
                if shared_authors:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
                    continue
                
                # Check Category overlap + Text Similarity
                shared_categories = set(paper['categories']).intersection(set(other['categories']))
                if shared_categories:
                    # Calculate Cosine Similarity of Dense Embeddings
                    cos_sim = float(np.dot(X[i], X[j]))
                    if cos_sim >= sim_threshold:
                        A[i, j] = 1.0
                        A[j, i] = 1.0
                        
        return A, X

    def _build_tfidf_features(self, texts: List[str], max_features: int) -> np.ndarray:
        """
        Builds L2-normalized TF-IDF dense feature vectors from text strings from scratch.
        (Deprecated in favor of semantic embeddings, kept as fallback)
        """
        # Tokenize and clean
        tokenized_texts: List[List[str]] = []
        word_pattern = re.compile(r'\b[a-zA-Z]{3,15}\b')
        for text in texts:
            tokens = word_pattern.findall(text.lower())
            tokenized_texts.append(tokens)

        # Compute document frequencies
        df = Counter()
        for tokens in tokenized_texts:
            for token in set(tokens):
                df[token] += 1

        # Drop standard stopwords & select top vocabulary
        stopwords = {
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'our', 'are', 'was', 
            'were', 'been', 'have', 'has', 'this', 'these', 'those', 'their', 'they', 
            'them', 'can', 'will', 'not', 'but', 'more', 'about', 'one', 'two', 'use', 
            'used', 'using', 'which', 'its', 'also', 'such', 'into', 'than', 'only'
        }
        
        vocab_candidates = [
            token for token, count in df.most_common(max_features + len(stopwords))
            if token not in stopwords
        ]
        num_docs = len(texts)
        vocab = vocab_candidates[:max_features]
        vocab_idx = {token: idx for idx, token in enumerate(vocab)}

        # Store vocab mapping and IDF values for later query vectorization
        self.vocab_idx = vocab_idx
        self.idf = {}
        for token in vocab:
            self.idf[token] = math.log((1 + num_docs) / (1 + df[token])) + 1.0

        # Fill TF-IDF matrix
        features = np.zeros((num_docs, len(vocab)))
        
        for i, tokens in enumerate(tokenized_texts):
            if not tokens:
                continue
            tf = Counter(tokens)
            doc_len = len(tokens)
            for token, count in tf.items():
                if token in vocab_idx:
                    idx = vocab_idx[token]
                    tf_val = count / doc_len
                    idf_val = self.idf[token]
                    features[i, idx] = tf_val * idf_val

        # L2 normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        features = features / norms
        return features

    def vectorize_query(self, query: str, instruction: str = "") -> np.ndarray:
        """
        Vectorizes a new search query using the semantic embedding model, or falls back to TF-IDF.
        Prepend instruction if provided for asymmetric search models like BGE-Large.
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            
            # Format query for asymmetric instruction-tuned model
            formatted_query = f"{instruction} {query}".strip()
            
            # Encode and return dense semantic vector
            q_emb = model.encode([formatted_query], convert_to_numpy=True)[0]
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm
            return q_emb
        except ImportError:
            # Fallback to TF-IDF
            if not hasattr(self, 'vocab_idx') or not hasattr(self, 'idf'):
                raise ValueError("Pipeline must build graph (and calculate TF-IDF) before vectorizing queries.")
                
            word_pattern = re.compile(r'\b[a-zA-Z]{3,15}\b')
            tokens = word_pattern.findall(query.lower())
            
            features = np.zeros(len(self.vocab_idx))
            if not tokens:
                return features
                
            tf = Counter(tokens)
            doc_len = len(tokens)
            
            for token, count in tf.items():
                if token in self.vocab_idx:
                    idx = self.vocab_idx[token]
                    tf_val = count / doc_len
                    idf_val = self.idf[token]
                    features[idx] = tf_val * idf_val

            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            return features


def generate_mock_data(num_nodes: int = 50, max_features: int = 1024, pipeline: ArXivIngestionPipeline = None) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Generates synthetic paper metadata, adjacency matrix, and feature matrix
    for offline testing and recovery fallbacks.
    """
    topics = [
        "Graph Neural Networks", "Retrieval Augmented Generation",
        "Large Language Models", "Numerical Physics Simulations",
        "Bayesian Neural Networks", "Quantum Computing Algorithms"
    ]
    
    papers: List[Dict[str, Any]] = []
    np.random.seed(42)

    topic_words = {
        "Graph Neural Networks": "graph nodes edges message passing adjacency laplacian spectral convolution",
        "Retrieval Augmented Generation": "rag vector databases document embedding retrieve generation query retrieval",
        "Large Language Models": "llm transformer attention pretraining tokenizer model parameters decoder token",
        "Numerical Physics Simulations": "differential equations simulation runge kutta Euler numerical physics solver grid discrete",
        "Bayesian Neural Networks": "uncertainty posterior prior variational inference mcmc gaussian process probability bayes",
        "Quantum Computing Algorithms": "qubit superposition entanglement quantum gates circuit fourier transform hamiltonian"
    }

    for i in range(num_nodes):
        topic_idx = i % len(topics)
        topic = topics[topic_idx]
        arxiv_id = f"2606.{10000 + i}"
        
        title = f"Advances in {topic}: Methodological Framework {i}"
        
        # Build distinct summary content for distinct topics
        words = topic_words[topic]
        summary = (
            f"We present research on {topic} for scientific applications. "
            f"Our methodology focuses on {words}. "
            f"This framework connects closely with other methods in the literature."
        )
        
        # Inject chain-like citation links
        if i > 0:
            summary += f" We build on the concepts in arXiv:2606.{10000 + i - 1}."
            
        # Explicitly inject bridge links to create structural bottlenecks bridging communities
        if i == 12:
            summary += " We bridge methods from arXiv:2606.10002 and arXiv:2606.10035."
        elif i == 28:
            summary += " This combines approaches from arXiv:2606.10015 and arXiv:2606.10042."
            
        authors = [f"Author_{i}_A", f"Author_{i}_B"]
        # Add shared authors to create co-authorship links within topics
        if i >= 6:
            authors.append(f"Author_{i-6}_A")
            
        categories = ["cs.LG" if topic_idx == 0 else "cs.IR" if topic_idx == 1 else "cs.CL" if topic_idx == 2 else "physics.comp-ph" if topic_idx == 3 else "stat.ML" if topic_idx == 4 else "quant-ph"]
        
        papers.append({
            'id': arxiv_id,
            'title': title,
            'summary': summary,
            'authors': authors,
            'categories': categories,
            'raw_id': f"http://arxiv.org/abs/{arxiv_id}v1"
        })

    # Build A and X using pipeline logic
    if pipeline is None:
        pipeline = ArXivIngestionPipeline()
    A, X = pipeline.build_graph(papers, max_features=max_features, sim_threshold=0.15)
    
    return papers, A, X

def generate_hierarchical_mock_data(max_features: int = 1024, pipeline: Optional[ArXivIngestionPipeline] = None) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Generates exactly 3 connected graph nodes with synthetic tree structures 
    for offline unit testing of hierarchical graph RAG.
    """
    papers = []
    
    for i in range(3):
        paper = {
            'id': f"2606.0000{i}",
            'title': f"Hierarchical Quantum Computing {i}",
            'summary': f"This is a mock paper {i} on quantum computing.",
            'authors': [f"Author {i}"],
            'categories': ["quant-ph"],
            'raw_id': f"http://arxiv.org/abs/2606.0000{i}v1"
        }
        
        # Inject citations for connectivity: 1 cites 0, 2 cites 1
        if i > 0:
            paper['summary'] += f" We cite arXiv:2606.0000{i-1}."
            
        tree = {
            "metadata": paper.copy(),
            "branches": [
                {
                    "title": "Introduction",
                    "content": [{"type": "text", "text": f"Introduction to paper {i}. Quantum computing is hard."}]
                },
                {
                    "title": "Methodology",
                    "content": [
                        {"type": "text", "text": "We formulate the model as follows."},
                        {"type": "equation", "text": f"E = mc^{i}"}
                    ]
                },
                {
                    "title": "Conclusion",
                    "content": [{"type": "text", "text": f"In conclusion, model {i} works well."}]
                }
            ]
        }
        paper['tree'] = tree
        papers.append(paper)
        
    if pipeline is None:
        pipeline = ArXivIngestionPipeline()
    A, X = pipeline.build_graph(papers, max_features=max_features, sim_threshold=0.15)
    
    return papers, A, X

