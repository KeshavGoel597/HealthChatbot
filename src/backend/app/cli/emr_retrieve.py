#!/usr/bin/env python3
"""
CLI for full RAG pipeline: Query → Seed CUIs → Graph Expansion → EMR Retrieval.

Usage:
    python -m app.cli.emr_retrieve "my head hurts" --emr ../backend/data/patient101.json
    python -m app.cli.emr_retrieve "diabetes and blood sugar" --emr data/patient101.json
    python -m app.cli.emr_retrieve --interactive --emr data/patient101.json
    python -m app.cli.emr_retrieve "headache" --emr data/patient101.json --hide-prompt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.term_extractor import TermExtractor
from app.services.rag.graph import KnowledgeGraph
from app.services.rag.graph_expand import (
    DIAGNOSTIC_RELATIONS,
    group_by_hop,
)
from app.services.rag.emr_match import MatchedSection
from app.services.rag.pipeline import run_pipeline


# ── Terminal colors ───────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[36m"
    MAGENTA = "\033[35m"
    WHITE   = "\033[97m"
    RED     = "\033[31m"
    BLUE    = "\033[34m"

CATEGORY_COLORS = {
    "lab": C.CYAN,
    "symptom": C.YELLOW,
    "diagnosis": C.RED,
    "medicine": C.MAGENTA,
    "comorbidity": C.RED,
    "history": C.WHITE,
    "comment": C.DIM,
    "vitals": C.BLUE,
    "recommended_labs": C.CYAN,
    "demographics": C.DIM,
    "discharge": C.WHITE,
}


# ── Default paths ─────────────────────────────────────────────────────

def _default_path(filename: str) -> str:
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(backend_dir, "..", filename)


# ── Display functions ─────────────────────────────────────────────────

def print_extraction(extraction) -> None:
    print(f"\n  {C.BOLD}═══ Term Extraction ═══{C.RESET}")
    print(f"  Intent:     {C.CYAN}{extraction.intent}{C.RESET}")
    print(f"  Categories: {C.MAGENTA}{extraction.categories}{C.RESET}")
    print(f"  Terms:      {C.YELLOW}{extraction.terms}{C.RESET}\n")


def print_phase1(query: str, results: list[dict]) -> None:
    print(f"\n  {C.BOLD}═══ Phase 1: Seed CUI Extraction ═══{C.RESET}")
    print(f'  Query: {C.WHITE}"{query}"{C.RESET}\n')
    for r in results:
        print(
            f"  {C.GREEN}●{C.RESET}  "
            f"{C.BOLD}{r['cui']:<12}{C.RESET} "
            f"{r['name']:<40} "
            f"{C.DIM}{r['score']:.4f}{C.RESET}"
        )
    if not results:
        print(f"  {C.RED}(no CUIs above threshold){C.RESET}")
    print()


def print_phase2(expanded: list) -> None:
    groups = group_by_hop(expanded)
    total = len(expanded)

    print(f"  {C.BOLD}═══ Phase 2: Graph Expansion ═══{C.RESET}")
    parts = []
    for hop in sorted(groups.keys()):
        label = "seeds" if hop == 0 else f"hop-{hop}"
        count = len(groups[hop])
        parts.append(f"{count} {label}")
    print(f"  {C.DIM}{' + '.join(parts)} = {total} total CUIs{C.RESET}\n")


def print_phase3(matches: list[MatchedSection], get_name) -> None:
    print(f"  {C.BOLD}═══ Phase 3: EMR Retrieval ═══{C.RESET}")
    print(f"  {C.DIM}Matched {len(matches)} EMR sections{C.RESET}\n")

    if not matches:
        print(f"  {C.RED}(no matching sections found){C.RESET}\n")
        return

    for i, m in enumerate(matches, 1):
        s = m.section
        color = CATEGORY_COLORS.get(s.category, C.WHITE)
        tag = f"[{s.category}]"

        # Main line: rank, category tag, text
        print(
            f"  {C.BOLD}{i:>3}.{C.RESET} "
            f"{color}{tag:<18}{C.RESET} "
            f"{C.WHITE}{s.text}{C.RESET}"
        )

        # Detail line: value, date, matched CUIs
        details = []
        if s.value and s.category == "lab":
            details.append(f"value={s.value}")
        if s.date:
            details.append(f"date={s.date}")
        details.append(f"score={m.best_score:.3f}")

        cui_names = [f"{c} ({get_name(c)})" for c in m.matched_cuis[:3]]
        if len(m.matched_cuis) > 3:
            cui_names.append(f"+{len(m.matched_cuis) - 3} more")

        print(
            f"       {C.DIM}{', '.join(details)}{C.RESET}"
        )
        print(
            f"       {C.DIM}matched: {', '.join(cui_names)}{C.RESET}"
        )
        print()


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full RAG pipeline: Query → Seeds → Graph → EMR Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "my head hurts" --emr data/patient101.json
  %(prog)s "diabetes and blood sugar" --emr data/patient102.json
  %(prog)s "kidney function" --emr data/patient101.json --match-threshold 0.3
    %(prog)s --interactive --emr data/patient101.json
    %(prog)s "headache" --emr data/patient101.json --hide-prompt
        """,
    )
    parser.add_argument("query", nargs="?", default=None, help="Natural language query")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode: keep entering queries")
    parser.add_argument("--emr", required=True, help="Path to patient EMR JSON file")

    # Phase 1 params
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Phase 1 seed top-k (default: 10)")
    parser.add_argument("-t", "--threshold", type=float, default=0.7, help="Phase 1 min score (default: 0.7)")

    # Phase 2 params
    parser.add_argument("-d", "--depth", type=int, default=2, help="Graph expansion depth (default: 2)")
    parser.add_argument("--all-relations", action="store_true", help="Use all relation types")

    # Phase 3 params
    parser.add_argument("--match-top-k", type=int, default=20, help="FAISS candidates per EMR section (default: 20)")
    parser.add_argument("--match-threshold", type=float, default=0.5, help="Min cosine sim for EMR match (default: 0.5)")
    parser.add_argument("--dedup", action="store_true", default=True, help="Deduplicate EMR sections (default: on)")

    # Phase 4 params
    parser.add_argument("--show-prompt", dest="show_prompt", action="store_true", default=True, help="Show the assembled LLM prompt (default: on)")
    parser.add_argument("--hide-prompt", dest="show_prompt", action="store_false", help="Hide the assembled LLM prompt")

    # Resource paths
    parser.add_argument("--embeddings", type=str, default=None)
    parser.add_argument("--graph", type=str, default=None)

    return parser.parse_args()


def run_query(query: str, args: argparse.Namespace, index: EmbeddingIndex, graph: KnowledgeGraph, extractor: TermExtractor) -> None:
    t_total = time.time()

    # ── Run pipeline ──
    allowed = None if args.all_relations else DIAGNOSTIC_RELATIONS
    result = run_pipeline(
        query,
        args.emr,
        index,
        graph,
        extractor=extractor,
        seed_top_k=args.top_k,
        seed_threshold=args.threshold,
        graph_depth=args.depth,
        allowed_relations=allowed,
        match_top_k=args.match_top_k,
        match_threshold=args.match_threshold,
        dedup=args.dedup,
    )

    print(f"  {C.DIM}EMR: {result.total_sections} sections from {args.emr}{C.RESET}")

    # ── Display phases ──
    print_extraction(result.extraction)
    print_phase1(query, result.seed_cuis)
    if result.expanded_cuis:
        print_phase2(result.expanded_cuis)
    print_phase3(result.matches, index.get_name)

    # ── Phase 4: Prompt ──
    if args.show_prompt:
        print(f"  {C.BOLD}═══ Phase 4: Assembled Prompt ═══{C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        for line in result.system_prompt.split('\n'):
            print(f"  {C.WHITE}{line}{C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {C.DIM}User message:{C.RESET}  {C.WHITE}\"{query}\"{C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {C.DIM}Prompt length: {len(result.system_prompt)} chars{C.RESET}\n")

    # ── Summary ──
    total_ms = (time.time() - t_total) * 1000
    print(
        f"  {C.DIM}Pipeline: "
        f"{len(result.seed_cuis)} seeds → "
        f"{result.expanded_cui_count} expanded CUIs → "
        f"{len(result.matches)}/{result.total_sections} EMR sections matched "
        f"[total: {total_ms / 1000:.1f}s]{C.RESET}\n"
    )


def interactive_loop(args: argparse.Namespace, index: EmbeddingIndex, graph: KnowledgeGraph, extractor: TermExtractor) -> None:
    print("\nInteractive mode - type a query and press Enter. 'quit' or Ctrl-C to exit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        run_query(query, args, index, graph, extractor)


def main() -> None:
    args = parse_args()
    if not args.query and not args.interactive:
        print("Error: provide a query or use --interactive mode.\n")
        sys.exit(1)

    # ── Load resources once, then reuse in interactive mode ──
    print(f"\n  {C.DIM}Loading resources...{C.RESET}")
    t0 = time.time()

    emb_path = args.embeddings or _default_path("GraphModel_SNOMED_CUI_Embedding.pkl")
    graph_path = args.graph or _default_path("SNOMED_CUI_MAJID_Graph_wSelf.pkl")

    index = EmbeddingIndex(emb_path)
    graph = KnowledgeGraph(graph_path)
    extractor = TermExtractor()

    print(f"  {C.DIM}Resources loaded in {time.time() - t0:.1f}s{C.RESET}")

    if args.interactive:
        interactive_loop(args, index, graph, extractor)
        return

    run_query(args.query, args, index, graph, extractor)


if __name__ == "__main__":
    main()
