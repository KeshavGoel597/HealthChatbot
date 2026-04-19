# backend/scripts/benchmark.py
"""
Benchmark: RAG vs. EMR-summarize vs. raw-paste token usage.

Three modes per query:
  raw_paste     — Full EMR text pasted directly into the system prompt. No LLM pre-processing.
  emr_summarize — GeminiService calls extract_clinical_data() internally to structure the EMR first.
  rag           — 4-phase SNOMED RAG pipeline produces a focused system prompt.

Usage (from backend/ dir, with venv active):
    python -m scripts.benchmark \\
        --emr-path data/patient101.json \\
        --embedding-path ../../GraphModel_SNOMED_CUI_Embedding.pkl \\
        --graph-path ../../SNOMED_CUI_MAJID_Graph_wSelf.pkl

    # Custom queries:
    python -m scripts.benchmark \\
        --emr-path data/patient101.json \\
        --embedding-path ../../GraphModel_SNOMED_CUI_Embedding.pkl \\
        --graph-path ../../SNOMED_CUI_MAJID_Graph_wSelf.pkl \\
        --queries "What meds am I on?" "Any allergies?"

Requires GEMINI_API_KEY in backend/.env
Outputs a CSV to scripts/results/ and prints a summary table.
"""
import argparse
import asyncio
import contextlib
import csv
import io
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.getLogger("app.services.presidio_anonymizer").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from app.services.gemini_service import GeminiService, build_emr_system_prompt
from app.services.presidio_anonymizer import init_presidio_engines
from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.graph import KnowledgeGraph
from app.services.rag.term_extractor import TermExtractor
from app.services.rag.pipeline import run_pipeline


class C:
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    RED   = "\033[31m"
    GREEN = "\033[32m"
    BLUE  = "\033[34m"

COLOR = {"raw_paste": C.RED, "emr_summarize": C.GREEN, "rag": C.BLUE}
LABEL = {"raw_paste": "RAW",  "emr_summarize": "SUMM", "rag": "RAG"}
METRIC_KEYS = ("input_tokens", "output_tokens", "total_tokens", "latency_ms", "response_chars", "response")

QUERIES = [
    "What medications am I currently taking?",
    "What were my most recent lab results?",
    "Do I have any chronic conditions or comorbidities?",
    "What symptoms have I reported recently?",
    "What did my last doctor visit say?",
]

CSV_FIELDS = [
    "method", "question",
    "input_tokens", "output_tokens", "total_tokens",
    "latency_ms", "response_chars",
    "seed_cuis", "emr_matches", "context_chars",
    "response",
]


def _print_section(method: str, prompt: str, response: str) -> None:
    color = COLOR[method]
    label = LABEL[method]
    print(f"\n{color}{C.BOLD}{'━'*80}")
    print(f"  [{label}] PROMPT")
    print(f"{'━'*80}{C.RESET}")
    print(f"{color}{prompt}{C.RESET}")
    print(f"\n{color}{C.BOLD}{'━'*80}")
    print(f"  [{label}] RESPONSE")
    print(f"{'━'*80}{C.RESET}")
    print(f"{color}{response}{C.RESET}")


@contextlib.asynccontextmanager
async def _suppress_stdout():
    """Suppress stdout from internal service prints during LLM calls."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


async def _call_gemini(
    gemini: GeminiService,
    message: str,
    system_prompt: str,
    patient_id: str,
    emr_consent: bool,
    presidio_analyzer=None,
    presidio_anonymizer=None,
) -> dict:
    start = time.perf_counter()
    async with _suppress_stdout():
        result = await gemini.chat(
            message=message,
            patient_id=patient_id,
            emr_consent=emr_consent,
            system_prompt=system_prompt,
            presidio_analyzer=presidio_analyzer,
            presidio_anonymizer=presidio_anonymizer,
        )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return {
        "input_tokens":   result["input_tokens"],
        "output_tokens":  result["output_tokens"],
        "total_tokens":   result["total_tokens"],
        "latency_ms":     latency_ms,
        "response_chars": len(result["response"]),
        "response":       result["response"],
    }


async def run(args):
    patient_id = Path(args.emr_path).stem
    queries = args.queries if args.queries else QUERIES

    with open(args.emr_path, "r") as f:
        emr_text = f.read()
    raw_system_prompt = build_emr_system_prompt(emr_text)

    with contextlib.redirect_stdout(io.StringIO()):
        gemini = GeminiService()
        index = EmbeddingIndex(args.embedding_path)
        graph = KnowledgeGraph(args.graph_path)
        extractor = TermExtractor()

    presidio_analyzer, presidio_anonymizer = init_presidio_engines()

    rows = []

    for query in queries:
        # ── raw_paste ──
        raw = await _call_gemini(gemini, query, raw_system_prompt, patient_id, True, presidio_analyzer, presidio_anonymizer)
        _print_section("raw_paste", raw_system_prompt, raw["response"])

        # ── emr_summarize ──
        summ = await _call_gemini(gemini, query, "", patient_id, True, presidio_analyzer, presidio_anonymizer)
        _print_section("emr_summarize", "(GeminiService builds prompt internally via extract_clinical_data)", summ["response"])

        # ── rag ──
        seed_count = match_count = context_chars = 0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline = run_pipeline(
                    query=query,
                    emr_path=args.emr_path,
                    index=index,
                    graph=graph,
                    patient_id=patient_id,
                    extractor=extractor,
                )
            rag_prompt    = pipeline.system_prompt
            seed_count    = len(pipeline.seed_cuis)
            match_count   = len(pipeline.matches)
            context_chars = len(pipeline.context_text)
        except Exception as e:
            rag_prompt = ""

        rag = await _call_gemini(gemini, query, rag_prompt, patient_id, True, presidio_analyzer, presidio_anonymizer)
        _print_section("rag", rag_prompt, rag["response"])

        for method, result, stats in [
            ("raw_paste",     raw,  (0,          0,           0)),
            ("emr_summarize", summ, (0,          0,           0)),
            ("rag",           rag,  (seed_count, match_count, context_chars)),
        ]:
            rows.append({
                "method": method, "question": query,
                "seed_cuis": stats[0], "emr_matches": stats[1], "context_chars": stats[2],
                **{k: result[k] for k in METRIC_KEYS},
            })

    if not rows:
        return

    # ── Write CSV ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.output) / f"benchmark_{timestamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary table ──
    raw_rows  = [r for r in rows if r["method"] == "raw_paste"]
    summ_rows = [r for r in rows if r["method"] == "emr_summarize"]
    rag_rows  = [r for r in rows if r["method"] == "rag"]
    n = len(rag_rows)

    def avg(lst, key):
        return sum(r[key] for r in lst) / len(lst)

    raw_tot  = avg(raw_rows,  "total_tokens")
    summ_tot = avg(summ_rows, "total_tokens")
    rag_tot  = avg(rag_rows,  "total_tokens")

    summ_savings = 100 * (1 - summ_tot / raw_tot) if raw_tot else 0
    rag_savings  = 100 * (1 - rag_tot  / raw_tot) if raw_tot else 0

    R, G, B, BOLD, RST = C.RED, C.GREEN, C.BLUE, C.BOLD, C.RESET
    W = 63

    print(f"\n{BOLD}{'='*W}{RST}")
    print(f"{BOLD}  BENCHMARK SUMMARY  ({n} quer{'y' if n==1 else 'ies'} × 3 modes){RST}")
    print(f"{BOLD}{'='*W}{RST}")
    print(f"  {'':22}  {R}{BOLD}{'RAW':>7}{RST}   {G}{BOLD}{'SUMM':>7}{RST}   {B}{BOLD}{'RAG':>7}{RST}")
    print(f"  {'─'*22}  {'─'*7}   {'─'*7}   {'─'*7}")

    for label, key in [
        ("Avg input tokens",  "input_tokens"),
        ("Avg output tokens", "output_tokens"),
        ("Avg total tokens",  "total_tokens"),
        ("Avg latency (ms)",  "latency_ms"),
    ]:
        rv = avg(raw_rows, key); sv = avg(summ_rows, key); bv = avg(rag_rows, key)
        print(f"  {label:<22}  {R}{rv:>7.0f}{RST}   {G}{sv:>7.0f}{RST}   {B}{bv:>7.0f}{RST}")

    print(f"  {'─'*22}  {'─'*7}   {'─'*7}   {'─'*7}")
    print(f"  {'Token savings vs raw':<22}  {'—':>7}   {G}{summ_savings:>6.1f}%{RST}   {B}{rag_savings:>6.1f}%{RST}")
    print(f"{BOLD}{'='*W}{RST}")
    print(f"\n{C.DIM}Results saved → {out}{RST}\n")


def main():
    p = argparse.ArgumentParser(
        description="Benchmark raw-paste vs. EMR-summarize vs. RAG token usage"
    )
    p.add_argument("--emr-path",       required=True,  help="Path to patient EMR JSON")
    p.add_argument("--embedding-path", required=True,  help="Path to GraphModel_SNOMED_CUI_Embedding.pkl")
    p.add_argument("--graph-path",     required=True,  help="Path to SNOMED_CUI_MAJID_Graph_wSelf.pkl")
    p.add_argument("--output", default="scripts/results", help="Output directory for CSV")
    p.add_argument("--queries", nargs="+", default=None, help="Custom queries (default: built-in 5)")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
