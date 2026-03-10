"""
Shared EMR summary extraction utility.

Used by HuggingFaceService and MedGemmaService to extract key clinical
information from raw EMR strings via regex. Returns (summary, fields_used)
for GDPR Art. 15 evidence transparency.
"""

import re


def summarize_emr_context(raw_data: str) -> tuple[str, list]:
    """
    Extracts key clinical information from the raw EMR string.
    Returns (summary_text, list_of_field_names_used) for GDPR Art. 15.
    """
    summary = []
    fields_used = []

    age_match = re.search(r'age: "([^"]+)"', raw_data)
    sex_match = re.search(r'sex: "([^"]+)"', raw_data)
    if age_match or sex_match:
        age = age_match.group(1) if age_match else "?"
        sex = sex_match.group(1) if sex_match else "?"
        summary.append(f"PATIENT: Age {age}, Sex {sex}")
        fields_used.append("Patient Demographics")

    diagnoses = set(re.findall(r'"diag" => "([^"]+)"', raw_data))
    cleaned_diagnoses = [d.strip() for d in diagnoses if d.strip() and d.strip() != "@10"]
    if cleaned_diagnoses:
        summary.append(f"DIAGNOSES: {', '.join(cleaned_diagnoses)}")
        fields_used.append("Medical Diagnoses")

    symptoms = set(re.findall(r'"sym" => "([^"]+)"', raw_data))
    cleaned_symptoms = [s.strip() for s in symptoms if s.strip() and s.strip() != "FCU"]
    if cleaned_symptoms:
        summary.append(f"SYMPTOMS: {', '.join(cleaned_symptoms)}")
        fields_used.append("Recorded Symptoms")

    meds = set(re.findall(r'"medicine" => "([^"]+)"', raw_data))
    if meds:
        summary.append(f"MEDICATIONS: {', '.join(list(meds)[:10])}")
        fields_used.append("Prescribed Medications")

    lab_summary = []
    for lab in ["Hemoglobin", "RBS", "Total WBC Count", "Platelet Count"]:
        matches = re.findall(
            f'"name" => "{lab}", "value" => "([^"]+)", "date" => "([^"]+)"',
            raw_data,
        )
        if matches:
            last_val, last_date = matches[-1]
            lab_summary.append(f"{lab}: {last_val} ({last_date})")
    if lab_summary:
        summary.append(f"RECENT LABS: {', '.join(lab_summary)}")
        fields_used.append("Laboratory Results")

    return "\n".join(summary), fields_used
