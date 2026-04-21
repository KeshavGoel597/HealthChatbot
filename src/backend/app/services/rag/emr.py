"""
EMR parser and section extractor.

Parses Ruby-style JSON EMR files into structured sections.
Each section is a discrete, searchable piece of clinical information.
No torch/faiss/embedding dependencies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EMRSection:
    """A discrete piece of information from an EMR."""

    category: str  # "lab", "symptom", "diagnosis", "medicine", "comorbidity",
    #                 "history", "comment", "vitals", "recommended_labs"
    text: str  # Primary text content (searchable)
    date: str = ""  # Date of record
    value: str = ""  # Associated value (lab result, etc.)
    raw: dict = field(default_factory=dict)  # Original dict


def parse_emr_file(path: str) -> dict:
    """Parse a Ruby-style JSON EMR file into a Python dict.

    Handles:
        - 'Patient data: ' prefix
        - Ruby hash rockets ("key" => val)
        - Bare symbol keys (age: "69")
    """
    text = Path(path).read_text()

    # Strip prefix
    text = text.strip()
    if text.startswith("Patient data:"):
        text = text[len("Patient data:") :].strip()

    # "key" => val  →  "key": val
    text = re.sub(r'"(\w[^"]*?)"\s*=>\s*', r'"\1": ', text)

    # Bare symbol keys:  {age: "69"}  →  {"age": "69"}
    # Use alternation to skip quoted strings so we don't corrupt values
    # containing ", word:" patterns.
    def _quote_bare_key(m: re.Match) -> str:
        if m.group(1):  # quoted string — preserve as-is
            return m.group(0)
        return f' "{m.group(2)}":'

    text = re.sub(
        r'("(?:[^"\\]|\\.)*")'       # group 1: quoted string (skip)
        r'|(?<=[{,])\s*(\w+):',        # group 2: bare key (fix)
        _quote_bare_key,
        text,
    )

    return json.loads(text)


def extract_sections(emr: dict) -> list[EMRSection]:
    """Extract searchable sections from a parsed EMR dict.

    Returns flat list of EMRSection objects, one per discrete
    piece of clinical information.
    """
    sections: list[EMRSection] = []

    # ── Demographics ──
    if emr.get("age"):
        sections.append(EMRSection(category="demographics", text=f"Age: {emr['age']}"))
    if emr.get("sex"):
        sections.append(EMRSection(category="demographics", text=f"Sex: {emr['sex']}"))

    # ── Lab data ──
    for lab in emr.get("lab_data", []):
        sections.append(
            EMRSection(
                category="lab",
                text=lab.get("name", ""),
                date=lab.get("date", ""),
                value=lab.get("value", ""),
                raw=lab,
            )
        )

    # ── Prescriptions (mixed bag of clinical data) ──
    for rx in emr.get("prescriptions", []):
        # Medicine entries have a different shape
        if "medicine" in rx:
            sections.append(
                EMRSection(category="medicine", text=rx["medicine"], raw=rx)
            )
            continue

        name = rx.get("name", "")
        value = rx.get("value", "")
        date = rx.get("date", "")

        if name == "Symptoms" or name == "Reason for Admission/Symptoms & Clinical findings":
            # value is a list of {sym, dur, end}
            if isinstance(value, list):
                for sym in value:
                    sections.append(
                        EMRSection(
                            category="symptom",
                            text=sym.get("sym", ""),
                            date=date,
                            raw=sym,
                        )
                    )

        elif name == "Diagnosis":
            if isinstance(value, list):
                for diag in value:
                    sections.append(
                        EMRSection(
                            category="diagnosis",
                            text=diag.get("diag", ""),
                            date=date,
                            raw=diag,
                        )
                    )

        elif name == "Comorbidity":
            if isinstance(value, list):
                for comor in value:
                    text = comor.get("diag", "")
                    if text and not text.startswith("@"):  # skip "@10" etc.
                        sections.append(
                            EMRSection(
                                category="comorbidity",
                                text=text,
                                date=date,
                                raw=comor,
                            )
                        )

        elif name == "Patient History":
            if isinstance(value, str) and value.strip():
                sections.append(
                    EMRSection(
                        category="history",
                        text=value.replace("\r\n", " ").replace("●", "").strip(),
                        date=date,
                        raw=rx,
                    )
                )

        elif name == "Comments":
            if isinstance(value, str) and value.strip() and value.strip() != ".":
                sections.append(
                    EMRSection(
                        category="comment",
                        text=value.replace("\r\n", " ").strip(),
                        date=date,
                        raw=rx,
                    )
                )

        elif name == "RecommendedLabs":
            if isinstance(value, str) and value.strip():
                sections.append(
                    EMRSection(
                        category="recommended_labs",
                        text=value,
                        date=date,
                        raw=rx,
                    )
                )

        elif name in ("Systolic", "Diastolic", "Pulse"):
            if isinstance(value, str) and value.strip() and value.strip() != ".":
                sections.append(
                    EMRSection(
                        category="vitals",
                        text=f"{name}: {value}",
                        date=date,
                        raw=rx,
                    )
                )

    # ── Discharge summary ──
    for item in emr.get("discharge_summary", []):
        if isinstance(item, dict):
            text = item.get("summary", item.get("text", ""))
            if text:
                sections.append(
                    EMRSection(category="discharge", text=text, raw=item)
                )

    return sections


def deduplicate_sections(sections: list[EMRSection]) -> list[EMRSection]:
    """Remove duplicate sections (same category + text), keeping latest date."""
    seen: dict[tuple[str, str], EMRSection] = {}
    for s in sections:
        normalized = re.sub(r"\s+", " ", s.text.strip()).upper()
        key = (s.category, normalized)
        if key not in seen:
            seen[key] = s
        else:
            # Keep the one with the latest date (or first if no dates)
            if s.date and s.date > seen[key].date:
                seen[key] = s
    return list(seen.values())
