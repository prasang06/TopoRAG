import sys
import ollama
from typing import Dict, Any

from src.formatter import ContextFormatter

SYSTEM_PROMPT = """You are an expert research assistant specializing in computational physics and numerical methods.
Your task is to synthesize a rigorously cited, hallucination-free literature review based ONLY on the provided extracted context.

CRITICAL INSTRUCTIONS:
1. NARRATIVE STRUCTURE: You MUST create a separate, numbered step for EVERY SINGLE PAPER provided in the [EXTRACTED CONTEXT]. If there are 10 papers in the context, there must be 10 numbered steps. Do not skip any papers.
   First, cover the [STRUCTURAL BOTTLENECK] papers. 
   Then, cover the [SEMANTIC HIT] papers.
2. EXACT CITATION FORMAT: For each step, provide the paper link on a single line formatted EXACTLY like this:
   **Read:** [Paper Title Here](https://arxiv.org/abs/1234.5678)
3. NO INLINE CITATIONS: Do NOT mention the arXiv ID, theorem names, or section names anywhere else in the text. The ONLY place the paper should be cited or linked is in the "**Read:**" line.
4. NO OUTSIDE KNOWLEDGE: Base your entire synthesis ONLY on the provided context. If the context is insufficient, state that."""

CURRICULUM_PROMPT = """You are an expert research advisor specializing in computational physics and numerical methods.
Your task is to generate a personalized, sequential reading curriculum for a student embarking on a new research project, based ONLY on the provided extracted context.

CRITICAL INSTRUCTIONS:
1. SEQUENTIAL SYLLABUS: You MUST create a separate, numbered step for EVERY SINGLE PAPER provided in the [EXTRACTED CONTEXT]. If there are 10 papers in the context, there must be 10 numbered steps in your syllabus. Do not skip any papers.
   - Start with the [STRUCTURAL BOTTLENECK] papers. Explain WHY the student must read this first.
   - Move to the [SEMANTIC HIT] papers. Explain WHAT they will gain from reading it.
2. EXACT CITATION FORMAT: For each step, provide the paper link on a single line formatted EXACTLY like this:
   **Read:** [Paper Title Here](https://arxiv.org/abs/1234.5678)
3. NO INLINE CITATIONS: Do NOT mention the arXiv ID, theorem names, or section names anywhere else in the text. The ONLY place the paper should be cited or linked is in the "**Read:**" line.
4. NO OUTSIDE KNOWLEDGE: Base your syllabus ONLY on the provided context."""

class SemanticSynthesizer:
    """
    Handles the connection to the local Ollama LLM to generate the final literature review
    using the formatted hierarchical RAG context.
    """
    def __init__(self, model: str = "gemma4:31b-cloud"):
        self.model = model
        
    def extract_arxiv_query(self, user_query: str) -> str:
        """
        Uses the LLM to parse a natural language query into a syntactically correct arXiv boolean search string.
        """
        system_prompt = """You are an expert physics research librarian. 
Your task is to take a user's natural language research question and extract 1 to 3 core concepts, formatting them into an arXiv API search string.

RULES:
1. Output ONLY the raw search string, nothing else. No preamble, no explanation, no backticks.
2. Format the string using 'all:' prefix and 'AND' boolean operators.
3. If a concept is multiple words, enclose that specific concept in quotes.
4. Keep the concepts broad enough to find papers but specific enough to be relevant. Fix any obvious spelling mistakes in the user's query.

EXAMPLE INPUT: i want to learn about how quantum error correction helps in encoding data into qubits and performing operations leading into arithemetic
EXAMPLE OUTPUT: all:"quantum error correction" AND all:qubit AND all:arithmetic
"""
        print(f"[Synthesizer] Asking Ollama ({self.model}) to extract arXiv search tags...")
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"USER QUERY: {user_query}"}
                ]
            )
            query = response['message']['content'].strip()
            
            # Strip markdown code blocks if the LLM disobeyed
            if query.startswith("```") and query.endswith("```"):
                query = query.split("\n")[1:-1]
                query = "".join(query).strip()
            if query.startswith("`") and query.endswith("`"):
                query = query[1:-1].strip()
                
            if "all:" not in query:
                print(f"[Synthesizer] Warning: LLM produced invalid query format: {query}. Falling back to default.")
                return 'all:"quantum"'
                
            print(f"[Synthesizer] Extracted optimal arXiv query: {query}")
            return query
        except Exception as e:
            print(f"[Synthesizer] Failed to extract arXiv query: {e}. Falling back to default.")
            return 'all:"quantum"'
        
    def generate_synthesis(self, query: str, formatted_context: str, mode: str = "review") -> str:
        """
        Calls the Ollama chat API to stream the synthesis to the terminal in real-time.
        """
        if mode == "curriculum":
            system_prompt = CURRICULUM_PROMPT
            task_desc = "reading curriculum"
            print_title = "GENERATING RESEARCH CURRICULUM"
        else:
            system_prompt = SYSTEM_PROMPT
            task_desc = "literature review"
            print_title = "GENERATING LITERATURE REVIEW"

        user_prompt = f"USER QUERY / PROJECT IDEA: {query}\n\n=== EXTRACTED CONTEXT ===\n{formatted_context}\n\nPlease generate the {task_desc} now."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("\n" + "="*70)
        print(f"                 {print_title}                 ")
        print("="*70 + "\n")
        
        full_response = ""
        try:
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_response += content
                
            print("\n\n" + "="*70)
            return full_response
            
        except ConnectionError:
            error_msg = "[ERROR] Could not connect to the local Ollama daemon. Please ensure 'ollama serve' is running."
            print(error_msg)
            return error_msg
        except ollama.ResponseError as e:
            error_msg = f"[ERROR] Ollama returned an error: {e}. Please ensure the model '{self.model}' is installed via 'ollama pull {self.model}'."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"[ERROR] An unexpected error occurred: {e}"
            print(error_msg)
            return error_msg

if __name__ == "__main__":
    # Mock Phase 1 JSON Payload
    mock_payload = {
        "metrics": {"semantic_hits": 1, "structural_bridges": 1},
        "nodes": [
            {
                "idx": 0,
                "id": "1905.12345",
                "title": "Foundations of Quantum Error Correction",
                "authors": ["Alice Smith", "Bob Jones"],
                "role": "Structural Bridge",
                "relevant_branches": [
                    {
                        "title": "Theoretical Framework",
                        "content": [
                            {"type": "text", "text": "The basis of our framework relies on the stabilizer formalism."},
                            {"type": "equation", "text": "S_i |\\psi\\rangle = |\\psi\\rangle \\quad \\forall i"}
                        ]
                    }
                ]
            },
            {
                "idx": 1,
                "id": "2401.98765",
                "title": "Surface Codes in Superconducting Qubits",
                "authors": ["Charlie Brown"],
                "role": "Semantic Hit",
                "relevant_branches": [
                    {
                        "title": "Experimental Setup",
                        "content": [
                            {"type": "text", "text": "We apply the stabilizer framework directly to our 2D grid of superconducting qubits."}
                        ]
                    }
                ]
            }
        ]
    }
    
    print("[Testing] Formatting Mock Payload...")
    formatter = ContextFormatter()
    formatted_ctx = formatter.format_payload(mock_payload)
    
    print("[Testing] Invoking Synthesizer with model 'gemma4:31b-cloud'...")
    synth = SemanticSynthesizer(model="gemma4:31b-cloud")
    synth.generate_synthesis(query="quantum error correction surface codes", formatted_context=formatted_ctx, mode="curriculum")
