import logging
from typing import Any

from fastapi import Request

from app.services.llm_factory import get_llm_service
from app.services.rag_orchestrator import build_rag_system_prompt

logger = logging.getLogger(__name__)


def _resolve_emr_fields(llm_result: dict[str, Any], emr_fields_used_from_rag: list[str]) -> list[str]:
    emr_fields_used = llm_result.get("emr_fields_used", [])
    if emr_fields_used == ["RAG Pipeline (SNOMED Knowledge Graph)"] and emr_fields_used_from_rag:
        return emr_fields_used_from_rag
    return emr_fields_used


async def run_llm_turn(
    *,
    req: Request,
    message: str,
    patient_id: str,
    model_name: str | None,
    emr_consent: bool,
    history: list[dict[str, str]] | None = None,
    compacted_summary: str | None = None,
) -> dict[str, Any]:
    """Run a single provider turn with shared RAG/anonymization orchestration."""
    system_prompt, emr_fields_used_from_rag = build_rag_system_prompt(
        req=req,
        message=message,
        patient_id=patient_id,
        emr_consent=emr_consent,
    )

    print("=== SYSTEM PROMPT SENT TO LLM ===", flush=True)
    print(system_prompt if system_prompt else "(empty - no consent or no EMR)", flush=True)
    print("=== END SYSTEM PROMPT ===", flush=True)

    presidio_analyzer = getattr(req.app.state, "presidio_analyzer", None)
    presidio_anonymizer = getattr(req.app.state, "presidio_anonymizer", None)

    llm_service = get_llm_service(model_name)
    llm_result = await llm_service.chat(
        message,
        patient_id,
        history=history or [],
        compacted_summary=compacted_summary,
        system_prompt=system_prompt,
        emr_consent=emr_consent,
        presidio_analyzer=presidio_analyzer,
        presidio_anonymizer=presidio_anonymizer,
    )

    llm_result["emr_fields_used"] = _resolve_emr_fields(llm_result, emr_fields_used_from_rag)
    return llm_result
