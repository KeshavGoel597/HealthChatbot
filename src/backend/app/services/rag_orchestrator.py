import os
from typing import Tuple, List
from fastapi import Request
from app.services.rag.pipeline import run_pipeline

def get_emr_path(patient_id: str) -> str:
    """Resolve patient EMR file path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", f"{patient_id}.json")

def build_rag_system_prompt(
    req: Request,
    message: str,
    patient_id: str,
    emr_consent: bool,
) -> Tuple[str, List[str]]:
    """
    Executes the RAG pipeline if consent is given.
    Returns:
        system_prompt (str): The clinical context system prompt.
        emr_fields_used (List[str]): Evidence tracking list for GDPR Art 15.
    """
    system_prompt = ""
    emr_fields_used = []

    if emr_consent:
        try:
            index = getattr(req.app.state, "embedding_index", None)
            graph = getattr(req.app.state, "knowledge_graph", None)
            extractor = getattr(req.app.state, "term_extractor", None)
            emr_path = get_emr_path(patient_id)

            if index and graph:
                pipeline_result = run_pipeline(
                    query=message,
                    emr_path=emr_path,
                    index=index,
                    graph=graph,
                    patient_id=patient_id,
                    extractor=extractor,
                )
                system_prompt = pipeline_result.system_prompt
                
                for match in pipeline_result.matches:
                    category = match.section.category.title()
                    if match.section.text:
                        short_text = match.section.text[:50] + ("..." if len(match.section.text) > 50 else "")
                        category += f" ({short_text})"
                    if category not in emr_fields_used:
                        emr_fields_used.append(category)
        except Exception as e:
            print(f"RAG pipeline error: {e}")
            system_prompt = ""

    return system_prompt, emr_fields_used
