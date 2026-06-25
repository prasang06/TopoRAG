from typing import Dict, Any, List

class ContextFormatter:
    """
    Utility class to unroll the hierarchical JSON payload from Phase 1
    into an optimized, token-efficient string for the LLM.
    """
    
    @staticmethod
    def format_payload(payload: Dict[str, Any]) -> str:
        if not payload or "nodes" not in payload:
            return ""
            
        formatted_context = []
        
        # Sort nodes: Bottlenecks first, Semantic Hits second
        # Let's assume role is either "Semantic Hit" or "Structural Bridge"
        def sort_key(node):
            role = node.get("role", "").lower()
            return 0 if "bridge" in role or "bottleneck" in role else 1
            
        nodes = sorted(payload["nodes"], key=sort_key)
        
        for node in nodes:
            role = node.get("role", "")
            if "bridge" in role.lower() or "bottleneck" in role.lower():
                tag = "[STRUCTURAL BOTTLENECK]"
            else:
                tag = "[SEMANTIC HIT]"
                
            formatted_context.append(f"=== {tag} ===")
            formatted_context.append(f"Paper: {node.get('title', 'Unknown')} (URL: https://arxiv.org/abs/{node.get('id', 'Unknown')})")
            
            authors = node.get("authors", [])
            if authors:
                formatted_context.append(f"Authors: {', '.join(authors)}")
                
            branches = node.get("relevant_branches", [])
            if not branches:
                formatted_context.append("No specific branches extracted.\n")
                continue
                
            for branch in branches:
                formatted_context.append(f"\n--- Section: {branch.get('title', 'Unknown')} ---")
                
                for item in branch.get("content", []):
                    content_type = item.get("type", "")
                    text = item.get("text", "")
                    
                    if content_type == "equation":
                        formatted_context.append(f"[EXTRACTED LEAF: EQUATION]\n{text}")
                    elif content_type == "subsection":
                        formatted_context.append(f"[EXTRACTED LEAF: SUBSECTION]\n{text}")
                    else:
                        formatted_context.append(f"[EXTRACTED LEAF: TEXT]\n{text}")
            
            formatted_context.append("\n")
            
        return "\n".join(formatted_context)
