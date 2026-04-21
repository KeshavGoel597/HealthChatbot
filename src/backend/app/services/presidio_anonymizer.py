import logging
import re
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)

def init_presidio_engines() -> tuple[Any, Any]:
    """Initialize Presidio analyzer/anonymizer engines, tolerating missing deps.

    Returns (analyzer, anonymizer); either may be None if its package failed to import.
    """
    analyzer = None
    anonymizer = None
    try:
        from presidio_anonymizer import AnonymizerEngine
        anonymizer = AnonymizerEngine()
    except Exception as exc:
        logger.warning("Presidio anonymizer initialization failed. De-identification will be skipped: %s", exc)
    try:
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
    except Exception as exc:
        logger.warning("Presidio analyzer initialization failed. Falling back to limited de-identification: %s", exc)
    return analyzer, anonymizer


DEFAULT_PII_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "URL",
    "IP_ADDRESS",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IN_PAN",
    "IN_AADHAAR",
]

_NAME_INTRO_PATTERN = re.compile(
    r"\b(?:my\s+name\s+is|i\s+am|i'm)\s+([A-Za-z][A-Za-z\-' ]{1,80})",
    flags=re.IGNORECASE,
)


def _fallback_analyzer_results(text: str) -> list:
    """Create limited Presidio-compatible results when AnalyzerEngine is unavailable."""
    try:
        from presidio_anonymizer.entities import RecognizerResult
    except Exception:
        return []

    results = []
    for match in _NAME_INTRO_PATTERN.finditer(text):
        start, end = match.span(1)
        results.append(
            RecognizerResult(
                entity_type="PERSON",
                start=start,
                end=end,
                score=0.8,
            )
        )
    return results


def _regex_only_fallback_anonymize(text: str) -> str:
    """Fallback anonymization path when Presidio anonymizer is unavailable."""
    if not text:
        return text

    def _replace_name(match: re.Match) -> str:
        prefix = match.group(0)[: match.start(1) - match.start(0)]
        return f"{prefix}<PERSON>"

    return _NAME_INTRO_PATTERN.sub(_replace_name, text)


def _merge_results(primary: list, secondary: list) -> list:
    merged = []
    seen = set()
    for item in (primary or []) + (secondary or []):
        if isinstance(item, dict):
            entity_type = item.get("entity_type")
            start = item.get("start")
            end = item.get("end")
        else:
            entity_type = getattr(item, "entity_type", None)
            start = getattr(item, "start", None)
            end = getattr(item, "end", None)

        key = (entity_type, start, end)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _count_regex_fallback_replacements(text: str) -> int:
    return sum(1 for _ in _NAME_INTRO_PATTERN.finditer(text or ""))


def count_anonymized_replacements(
    text: str,
    analyzer: Any,
    anonymizer: Any,
    *,
    language: str = "en",
    entities: Iterable[str] = DEFAULT_PII_ENTITIES,
) -> int:
    """Return number of detected/anonymized entities for observability."""
    if not text:
        return 0

    if anonymizer is None:
        return _count_regex_fallback_replacements(text)

    try:
        if analyzer is not None:
            primary_results = analyzer.analyze(
                text=text,
                language=language,
                entities=list(entities),
            )
        else:
            primary_results = []
        fallback_results = _fallback_analyzer_results(text)
        return len(_merge_results(primary_results, fallback_results))
    except Exception:
        return 0


def anonymize_text_for_llm(
    text: str,
    analyzer: Any,
    anonymizer: Any,
    *,
    language: str = "en",
    entities: Iterable[str] = DEFAULT_PII_ENTITIES,
) -> str:
    """Anonymize PII in text using Presidio Analyzer + Anonymizer.

    If Presidio engines are unavailable or fail, the original text is returned.
    """
    if not text:
        return text

    if anonymizer is None:
        logger.warning("Presidio anonymizer unavailable; using regex-only fallback (limited PII coverage) for this request.")
        return _regex_only_fallback_anonymize(text)

    try:
        if analyzer is not None:
            primary_results = analyzer.analyze(
                text=text,
                language=language,
                entities=list(entities),
            )
        else:
            primary_results = []
        fallback_results = _fallback_analyzer_results(text)
        analyzer_results = _merge_results(primary_results, fallback_results)
        if not analyzer_results:
            return text

        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
        )
        return anonymized.text
    except Exception as exc:
        logger.warning("Presidio anonymization failed; continuing with original text: %s", exc)
        return text


def anonymize_history_for_llm(history: List[Dict[str, str]], analyzer: Any, anonymizer: Any) -> List[Dict[str, str]]:
    """Return a copy of history with anonymized content for outbound LLM usage."""
    sanitized: List[Dict[str, str]] = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        sanitized.append(
            {
                "role": role,
                "content": anonymize_text_for_llm(content, analyzer, anonymizer),
            }
        )
    return sanitized
