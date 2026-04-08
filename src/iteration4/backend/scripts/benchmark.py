# backend/scripts/benchmark.py
"""
Benchmark: RAG vs. full-context token usage and latency.

Usage (from backend/ dir, with venv active):
    python -m scripts.benchmark \\
        --emr-path data/patient101.json \\
        --embedding-path ../../GraphModel_SNOMED_CUI_Embedding.pkl \\
        --graph-path ../../SNOMED_CUI_MAJID_Graph_wSelf.pkl

Requires GEMINI_API_KEY in backend/.env
Outputs a CSV to scripts/results/ and prints a summary table.
"""
import argparse
import asyncio
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running as `python -m scripts.benchmark` from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.services.gemini_service import GeminiService
from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.graph import KnowledgeGraph
from app.services.rag.term_extractor import TermExtractor
from app.services.rag.pipeline import run_pipeline

QUERIES = [
    "What medications am I currently taking?",
    "What were my most recent lab results?",
    "Do I have any chronic conditions or comorbidities?",
    "What symptoms have I reported recently?",
    "What did my last doctor visit say?",
]


async def _call_gemini(
    gemini: GeminiService,
    message: str,
    system_prompt: str,
    patient_id: str,
    emr_consent: bool,
) -> dict:
    start = time.perf_counter()
    result = await gemini.chat(
        message=message,
        patient_id=patient_id,
        emr_consent=emr_consent,
        system_prompt=system_prompt,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return {
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "total_tokens": result["total_tokens"],
        "latency_ms": latency_ms,
        "response_chars": len(result["response"]),
    }


async def run(args):
    patient_id = Path(args.emr_path).stem  # e.g. "patient101"

    print("Loading Gemini service...")
    gemini = GeminiService()

    print("Loading RAG resources (this may take a minute)...")
    index = EmbeddingIndex(args.embedding_path)
    graph = KnowledgeGraph(args.graph_path)
    extractor = TermExtractor()

    rows = []

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] {query!r}")

        # ── RAG mode ──
        try:
            pipeline = run_pipeline(
                query=query,
                emr_path=args.emr_path,
                index=index,
                graph=graph,
                patient_id=patient_id,
                extractor=extractor,
            )
            rag_prompt = pipeline.system_prompt
            seed_count = len(pipeline.seed_cuis)
            match_count = len(pipeline.matches)
            context_chars = len(pipeline.context_text)
        except Exception as e:
            print(f"  RAG pipeline error: {e}")
            rag_prompt, seed_count, match_count, context_chars = "", 0, 0, 0

        rag = await _call_gemini(gemini, query, rag_prompt, patient_id, True)
        print(
            f"  [RAG]  tokens={rag['total_tokens']:>5}  "
            f"latency={rag['latency_ms']:>7}ms  "
            f"seeds={seed_count}  matches={match_count}  "
            f"context_chars={context_chars}"
        )

        # ── Full-context mode ──
        # system_prompt="" + emr_consent=True causes GeminiService to run
        # extract_clinical_data() internally (an extra API call not timed here).
        # This measures final LLM tokens only; full latency is understated vs RAG.
        full = await _call_gemini(gemini, query, "", patient_id, True)
        print(f"  [FULL] tokens={full['total_tokens']:>5}  latency={full['latency_ms']:>7}ms")

        rows.append({
            "query": query,
            "mode": "rag",
            "seed_cuis": seed_count,
            "emr_matches": match_count,
            "context_chars": context_chars,
            **rag,
        })
        rows.append({
            "query": query,
            "mode": "full_context",
            "seed_cuis": 0,
            "emr_matches": 0,
            "context_chars": 0,
            **full,
        })

    if not rows:
        print("No results collected. Exiting.")
        return

    # ── Write CSV ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.output) / f"benchmark_{timestamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {out}")

    # ── Print summary ──
    rag_rows  = [r for r in rows if r["mode"] == "rag"]
    full_rows = [r for r in rows if r["mode"] == "full_context"]
    n = len(rag_rows)

    def avg(lst, key):
        return sum(r[key] for r in lst) / len(lst)

    rag_tok  = avg(rag_rows,  "total_tokens")
    full_tok = avg(full_rows, "total_tokens")
    rag_lat  = avg(rag_rows,  "latency_ms")
    full_lat = avg(full_rows, "latency_ms")
    savings  = 100 * (1 - rag_tok / full_tok) if full_tok else 0

    print(f"\n{'='*55}")
    print(f"  BENCHMARK SUMMARY  ({n} queries × 2 modes)")
    print(f"{'='*55}")
    print(
        f"  Avg total tokens  "
        f"RAG: {rag_tok:>7.0f}   FULL: {full_tok:>7.0f}   "
        f"savings: {savings:.1f}%"
    )
    print(f"  Avg latency (ms)  RAG: {rag_lat:>7.0f}   FULL: {full_lat:>7.0f}")
    print(f"{'='*55}\n")


def main():
    p = argparse.ArgumentParser(
        description="Benchmark RAG vs full-context token usage"
    )
    p.add_argument("--emr-path",       required=True,  help="Path to patient EMR JSON")
    p.add_argument("--embedding-path", required=True,  help="Path to GraphModel_SNOMED_CUI_Embedding.pkl")
    p.add_argument("--graph-path",     required=True,  help="Path to SNOMED_CUI_MAJID_Graph_wSelf.pkl")
    p.add_argument(
        "--output",
        default="scripts/results",
        help="Output directory for CSV (default: scripts/results)",
    )
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
