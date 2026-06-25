import aiohttp
import tarfile
import io
import logging
import re
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

async def fetch_arxiv_eprint(arxiv_id: str) -> Optional[str]:
    """
    Asynchronously fetches the raw LaTeX e-print tarball from arXiv and extracts the main .tex file.
    Uses local caching to prevent rate limiting.
    Returns the raw LaTeX string, or None if it fails.
    """
    cache_dir = ".cache/arxiv_eprint"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{arxiv_id.replace('/', '_')}.tex")
    
    # Check cache first
    if os.path.exists(cache_file):
        logger.info(f"Cache hit for {arxiv_id}")
        with open(cache_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    url = f"https://export.arxiv.org/e-print/{arxiv_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as response:
                if response.status != 200:
                    logger.warning(f"Failed to download e-print for {arxiv_id}, status {response.status}")
                    return None
                
                content = await response.read()
                
                tex_string = None
                try:
                    # Attempt to extract tar.gz
                    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                        # Find the first .tex file
                        for member in tar.getmembers():
                            if member.name.endswith(".tex"):
                                f = tar.extractfile(member)
                                if f is not None:
                                    tex_string = f.read().decode('utf-8', errors='ignore')
                                    break
                except tarfile.TarError:
                    pass
                
                if tex_string is None:
                    tex_string = content.decode('utf-8', errors='ignore')
                
                # Save to cache
                if tex_string:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        f.write(tex_string)
                        
                return tex_string
                
    except Exception as e:
        logger.error(f"Error fetching e-print for {arxiv_id}: {e}")
        return None

def parse_latex_to_tree(tex_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses raw LaTeX into a Hierarchical Python Dictionary using robust regex.
    """
    tree: Dict[str, Any] = {
        "metadata": metadata,
        "branches": []
    }
    
    if not tex_content:
        return tree
        
    try:
        # Abstract extraction
        abs_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex_content, re.DOTALL)
        if abs_match:
            tree["metadata"]["abstract_parsed"] = abs_match.group(1).strip()
            
        # Split by sections
        sections = re.split(r'\\section\{([^}]*)\}', tex_content)
        
        for i in range(1, len(sections), 2):
            sec_title = sections[i].strip()
            sec_content = sections[i+1]
            
            branch = {"title": sec_title, "content": []}
            
            # Subsections
            subsecs = re.split(r'\\subsection\{([^}]*)\}', sec_content)
            
            def extract_content(text, parent_type=None, parent_title=None):
                eq_splits = re.split(r'\\begin\{equation\}(.*?)\\end\{equation\}', text, flags=re.DOTALL)
                for j in range(len(eq_splits)):
                    if j % 2 == 1:
                        branch["content"].append({"type": "equation", "text": eq_splits[j].strip()})
                    else:
                        txt = eq_splits[j].strip()
                        if txt:
                            if parent_type == "subsection" and parent_title:
                                branch["content"].append({"type": "subsection", "text": parent_title})
                                parent_title = None # Add only once
                            branch["content"].append({"type": "text", "text": txt})
            
            extract_content(subsecs[0])
            for j in range(1, len(subsecs), 2):
                extract_content(subsecs[j+1], "subsection", subsecs[j].strip())
                
            tree["branches"].append(branch)
            
    except Exception as e:
        logger.warning(f"Regex parser failed: {e}. Falling back to basic tree.")
        pass
        
    return tree

def generate_fallback_tree(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a simple tree structure from metadata if LaTeX parsing fails."""
    return {
        "metadata": metadata,
        "branches": [
            {
                "title": "Abstract",
                "content": [
                    {"type": "text", "text": metadata.get("abstract", "")}
                ]
            }
        ]
    }
