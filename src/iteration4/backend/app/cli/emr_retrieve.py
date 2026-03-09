#!/usr/bin/env python3
"""
CLI for full RAG pipeline: Query → Seed CUIs → Graph Expansion → EMR Retrieval.

Usage:
    python -m app.cli.emr_retrieve "my head hurts" --emr ../backend/data/patient101.json
    python -m app.cli.emr_retrieve "diabetes and blood sugar" --emr data/patient101.json
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.cui_search import find_cuis
from app.services.rag.term_extractor import TermExtractor
from app.services.rag.graph import KnowledgeGraph
from app.services.rag.graph_expand import (
    DIAGNOSTIC_RELATIONS,
    expand_cuis,
    group_by_hop,
)
from app.services.rag.emr import parse_emr_file, extract_sections, deduplicate_sections
from app.services.rag.emr_match import match_sections, MatchedSection
from app.services.rag.prompt import assemble_prompt


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


def print_phase2(expanded, get_name) -> None:
    groups = group_by_hop(expanded)
    seeds = len(groups.get(0, []))
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
        """,
    )
    parser.add_argument("query", help="Natural language query")
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
    parser.add_argument("--show-prompt", action="store_true", help="Show the assembled LLM prompt")

    # Resource paths
    parser.add_argument("--embeddings", type=str, default=None)
    parser.add_argument("--graph", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()

    # ── Load resources ──
    print(f"\n  {C.DIM}Loading resources...{C.RESET}")
    t0 = time.time()

    emb_path = args.embeddings or _default_path("GraphModel_SNOMED_CUI_Embedding.pkl")
    graph_path = args.graph or _default_path("SNOMED_CUI_MAJID_Graph_wSelf.pkl")

    index = EmbeddingIndex(emb_path)
    graph = KnowledgeGraph(graph_path)
    extractor = TermExtractor()

    print(f"  {C.DIM}Resources loaded in {time.time() - t0:.1f}s{C.RESET}")

    # ── Parse EMR ──
    emr = parse_emr_file(args.emr)
    sections = extract_sections(emr)
    if args.dedup:
        sections = deduplicate_sections(sections)
    print(f"  {C.DIM}EMR: {len(sections)} sections from {args.emr}{C.RESET}")

    # ── Phase 1: Seed CUIs ──
    t0 = time.time()
    phase1_results = find_cuis(args.query, index, top_k=args.top_k, threshold=args.threshold, extractor=extractor)
    print_phase1(args.query, phase1_results)

    seed_cuis = [r["cui"] for r in phase1_results]
    if not seed_cuis:
        print(f"  {C.RED}No seed CUIs — aborting.{C.RESET}")
        return

    # ── Phase 2: Graph expansion ──
    t0 = time.time()
    allowed = None if args.all_relations else DIAGNOSTIC_RELATIONS
    expanded = expand_cuis(seed_cuis, graph, depth=args.depth, allowed_relations=allowed)
    expanded_set = {r.cui for r in expanded}
    p2_ms = (time.time() - t0) * 1000

    print_phase2(expanded, index.get_name)

    # ── Phase 3: EMR matching ──
    t0 = time.time()
    matches = match_sections(
        sections,
        expanded_set,
        index,
        top_k=args.match_top_k,
        threshold=args.match_threshold,
    )
    p3_ms = (time.time() - t0) * 1000

    print_phase3(matches, index.get_name)

    # ── Phase 4: Prompt assembly ──
    if args.show_prompt:
        prompt = assemble_prompt(args.query, matches)
        print(f"  {C.BOLD}═══ Phase 4: Assembled Prompt ═══{C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        for line in prompt.split('\n'):
            print(f"  {C.WHITE}{line}{C.RESET}")
        print(f"  {C.DIM}{'─' * 60}{C.RESET}")
        print(f"  {C.DIM}Prompt length: {len(prompt)} chars{C.RESET}\n")

    # ── Summary ──
    total_ms = (time.time() - t_total) * 1000
    print(
        f"  {C.DIM}Pipeline: "
        f"{len(seed_cuis)} seeds → "
        f"{len(expanded_set)} expanded CUIs ({p2_ms:.0f}ms) → "
        f"{len(matches)}/{len(sections)} EMR sections matched ({p3_ms:.0f}ms) "
        f"[total: {total_ms / 1000:.1f}s]{C.RESET}\n"
    )


if __name__ == "__main__":
    main()
