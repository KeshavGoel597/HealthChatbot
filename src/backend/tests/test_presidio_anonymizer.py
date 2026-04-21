from app.services.presidio_anonymizer import (
    DEFAULT_PII_ENTITIES,
    anonymize_history_for_llm,
    anonymize_text_for_llm,
    count_anonymized_replacements,
)


class DummyResult:
    def __init__(self, text: str):
        self.text = text


class DummyRecognizerResult:
    def __init__(self, entity_type: str, start: int, end: int):
        self.entity_type = entity_type
        self.start = start
        self.end = end


class DummyAnalyzer:
    def __init__(self, has_pii: bool = True, raise_error: bool = False):
        self.has_pii = has_pii
        self.raise_error = raise_error

    def analyze(self, text, language, entities):
        if self.raise_error:
            raise RuntimeError("analyzer failed")
        if self.has_pii:
            return [DummyRecognizerResult(entity_type="PHONE_NUMBER", start=8, end=18)]
        return []


class DummyAnonymizer:
    def anonymize(self, text, analyzer_results):
        _ = analyzer_results
        return DummyResult(text.replace("9991112222", "<PHONE_NUMBER>"))


def test_anonymize_text_for_llm_masks_phone_number():
    analyzer = DummyAnalyzer(has_pii=True)
    anonymizer = DummyAnonymizer()

    text = "Call me at 9991112222"
    result = anonymize_text_for_llm(text, analyzer, anonymizer)

    assert "9991112222" not in result
    assert "<PHONE_NUMBER>" in result


def test_anonymize_text_for_llm_returns_original_when_no_findings():
    analyzer = DummyAnalyzer(has_pii=False)
    anonymizer = DummyAnonymizer()

    text = "No private details here"
    result = anonymize_text_for_llm(text, analyzer, anonymizer)

    assert result == text


def test_anonymize_text_for_llm_falls_back_on_error():
    analyzer = DummyAnalyzer(raise_error=True)
    anonymizer = DummyAnonymizer()

    text = "Call me at 9991112222"
    result = anonymize_text_for_llm(text, analyzer, anonymizer)

    assert result == text


def test_anonymize_history_for_llm_returns_sanitized_copy():
    analyzer = DummyAnalyzer(has_pii=True)
    anonymizer = DummyAnonymizer()
    history = [
        {"role": "user", "content": "My number is 9991112222"},
        {"role": "assistant", "content": "Thanks"},
    ]

    sanitized = anonymize_history_for_llm(history, analyzer, anonymizer)

    assert sanitized[0]["content"] == "My number is <PHONE_NUMBER>"
    assert history[0]["content"] == "My number is 9991112222"


def test_anonymize_text_for_llm_fallback_name_redaction_when_analyzer_missing():
    class FallbackAwareAnonymizer:
        def anonymize(self, text, analyzer_results):
            if analyzer_results:
                return DummyResult(text.replace("advait", "<PERSON>"))
            return DummyResult(text)

    text = "hi my name is advait what is my name"
    result = anonymize_text_for_llm(text, None, FallbackAwareAnonymizer())

    assert "advait" not in result.lower()
    assert "<PERSON>" in result


def test_count_anonymized_replacements_counts_regex_fallback_when_anonymizer_missing():
    text = "hi my name is advait"

    replacements = count_anonymized_replacements(text, analyzer=None, anonymizer=None)

    assert replacements == 1


def test_default_entities_use_supported_indian_aadhaar_spelling():
    assert "IN_AADHAAR" in DEFAULT_PII_ENTITIES
    assert "IN_AADHAR" not in DEFAULT_PII_ENTITIES
