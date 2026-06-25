import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.latex_parser import fetch_arxiv_eprint, parse_latex_to_tree, generate_fallback_tree
from src.retrieval import GraphRAGRetrievalEngine

MOCK_METADATA = {"id": "1234.5678", "abstract": "This is a mock abstract."}

@pytest.fixture
def retrieval_engine():
    return GraphRAGRetrievalEngine()

def test_parse_latex_success():
    mock_tex = r"""
    \begin{document}
    \begin{abstract}
    We present a new algorithm.
    \end{abstract}
    \section{Introduction}
    This is the intro.
    \subsection{Background}
    Some background.
    \section{Method}
    We use the following equation:
    \begin{equation}
    E = mc^2
    \end{equation}
    \end{document}
    """
    
    tree = parse_latex_to_tree(mock_tex, MOCK_METADATA.copy())
    
    assert "branches" in tree
    assert len(tree["branches"]) >= 2
    
    # Check intro
    intro = tree["branches"][0]
    assert intro["title"] == "Introduction"
    assert any(c["type"] == "text" and "intro" in c["text"] for c in intro["content"])
    assert any(c["type"] == "subsection" and "Background" in c["text"] for c in intro["content"])
    
    # Check method
    method = tree["branches"][1]
    assert method["title"] == "Method"
    assert any(c["type"] == "equation" and "E = mc^2" in c["text"] for c in method["content"])

def test_parse_latex_no_subsections():
    mock_tex = r"""
    \begin{document}
    \section{Introduction}
    Just a plain paragraph without any subsections.
    \end{document}
    """
    tree = parse_latex_to_tree(mock_tex, MOCK_METADATA.copy())
    
    assert len(tree["branches"]) == 1
    intro = tree["branches"][0]
    assert intro["title"] == "Introduction"
    
    # Ensure no subsections
    assert not any(c["type"] == "subsection" for c in intro["content"])
    # Ensure text is parsed
    assert any(c["type"] == "text" and "plain paragraph" in c["text"] for c in intro["content"])

@pytest.mark.asyncio
async def test_simulated_download_failure():
    # Patch aiohttp.ClientSession.get to raise an exception
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.side_effect = Exception("Network timeout")
        
        result = await fetch_arxiv_eprint("invalid_id")
        assert result is None
        
        # When result is None, pipeline will call generate_fallback_tree
        fallback = generate_fallback_tree(MOCK_METADATA.copy())
        
        assert "branches" in fallback
        assert len(fallback["branches"]) == 1
        assert fallback["branches"][0]["title"] == "Abstract"
        assert fallback["branches"][0]["content"][0]["text"] == "This is a mock abstract."

def test_micro_retrieval(retrieval_engine):
    mock_tree = {
        "metadata": MOCK_METADATA,
        "branches": [
            {
                "title": "Introduction",
                "content": [{"type": "text", "text": "This paper is about neural networks."}]
            },
            {
                "title": "Quantum Error Correction",
                "content": [{"type": "text", "text": "We apply quantum error correction using surface codes."}]
            }
        ]
    }
    
    query = "quantum error correction surface codes"
    
    # Micro retrieve
    results = retrieval_engine._micro_retrieve_tree(query, mock_tree, top_branches=1)
    
    assert len(results) == 1
    assert results[0]["title"] == "Quantum Error Correction"
    assert "surface codes" in results[0]["content"][0]["text"]
