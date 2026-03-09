#!/usr/bin/env python3
"""
CLI for testing Phase 1 → Phase 2 pipeline: Query → Seed CUIs → Graph Expansion.

Usage:
    # Full pipeline: query → seed CUIs → graph expansion
    python -m app.cli.graph_expand "my head hurts"

    # Skip Phase 1, pass CUIs directly
    python -m app.cli.graph_expand --cuis C0018681 C0008031

    # Custom depth and relation filtering
    python -m app.cli.graph_expand "chest pain" --depth 1 --max-per-hop 20
"""

import argparse
import os
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.cui_search import find_cuis
from app.services.rag.term_extractor import TermExtractor
from app.services.rag.graph import KnowledgeGraph
from app.services.rag.graph_expand import (
    DIAGNOSTIC_RELATIONS,
    ExpandedCUI,
    expand_cuis,
    group_by_hop,
)


# ── Terminal colors ───────────────────────────────────────────────────

class C:
    """ANSI color codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    CYAN    = "\033[36m"
    MAGENTA = "\033[35m"
    WHITE   = "\033[97m"
    RED     = "\033[31m"

HOP_COLORS = {0: C.GREEN, 1: C.YELLOW, 2: C.CYAN, 3: C.MAGENTA}


def _color(hop: int) -> str:
    return HOP_COLORS.get(hop, C.WHITE)


# ── Default paths ─────────────────────────────────────────────────────

def _default_path(filename: str) -> str:
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(backend_dir, "..", filename)


# ── Display functions ─────────────────────────────────────────────────

def print_phase1(query: str, results: list[dict]) -> None:
    """Print Phase 1 seed CUI results with color."""
    print(f"\n  {C.BOLD}═══ Phase 1: Seed CUI Extraction ═══{C.RESET}")
    print(f'  Query: {C.WHITE}"{query}"{C.RESET}\n')

    if not results:
        print(f"  {C.RED}(no CUIs above threshold){C.RESET}\n")
        return

    for i, r in enumerate(results, 1):
        print(
            f"  {_color(0)}●{C.RESET}  "
            f"{C.BOLD}{r['cui']:<12}{C.RESET} "
            f"{r['name']:<40} "
            f"{C.DIM}{r['score']:.4f}{C.RESET}"
        )
    print()


def print_expansion(
    results: list[ExpandedCUI],
    get_name: Callable[[str], str],
    max_display: int = 20,
) -> None:
    """Print Phase 2 expansion results with colored hop levels."""
    groups = group_by_hop(results)
    seeds = len(groups.get(0, []))
    total = len(results)

    print(f"  {C.BOLD}═══ Phase 2: Graph Expansion ═══{C.RESET}")
    parts = []
    for hop in sorted(groups.keys()):
        label = "seeds" if hop == 0 else f"hop-{hop}"
        parts.append(f"{_color(hop)}{len(groups[hop])} {label}{C.RESET}")
    print(f"  {' + '.join(parts)} = {C.BOLD}{total} total{C.RESET}\n")

    for hop in sorted(groups.keys()):
        if hop == 0:
            continue  # seeds already shown in Phase 1

        items = groups[hop]
        color = _color(hop)
        print(f"  {color}{C.BOLD}Hop {hop}{C.RESET} {C.DIM}({len(items)} CUIs){C.RESET}")

        shown = items[:max_display]
        for r in shown:
            name = get_name(r.cui)
            src_name = get_name(r.source) if r.source else ""
            print(
                f"    {color}├{C.RESET} "
                f"{C.BOLD}{r.cui:<12}{C.RESET} "
                f"{name:<35} "
                f"{C.DIM}← {src_name} [{r.relation}]{C.RESET}"
            )

        remaining = len(items) - len(shown)
        if remaining > 0:
            print(f"    {color}└ {C.DIM}... and {remaining} more{C.RESET}")
        print()


def print_summary(results: list[ExpandedCUI], elapsed_ms: float) -> None:
    """Print final summary line."""
    groups = group_by_hop(results)
    parts = []
    for hop in sorted(groups.keys()):
        label = "seeds" if hop == 0 else f"hop-{hop}"
        parts.append(f"{len(groups[hop])} {label}")
    summary = " + ".join(parts)
    print(
        f"  {C.DIM}Summary: {summary} = {len(results)} total CUIs "
        f"({elapsed_ms:.0f}ms){C.RESET}\n"
    )


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 → Phase 2 pipeline: Query → Seed CUIs → Graph Expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "my head hurts"
  %(prog)s "chest pain" --depth 1 --max-per-hop 20
  %(prog)s --cuis C0018681 C0008031
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("query", nargs="?", default=None, help="Natural language query")
    group.add_argument("--cuis", nargs="+", help="Skip Phase 1, pass CUI codes directly")

    parser.add_argument("-d", "--depth", type=int, default=2, help="BFS hop depth (default: 2)")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Phase 1 top-k seeds (default: 10)")
    parser.add_argument("-t", "--threshold", type=float, default=0.7, help="Phase 1 min score (default: 0.7)")
    parser.add_argument("--max-per-hop", type=int, default=None, help="Cap neighbors per source per hop")
    parser.add_argument("--max-display", type=int, default=20, help="Max CUIs to display per hop (default: 20)")
    parser.add_argument("--all-relations", action="store_true", help="Use all 108 relations (no filter)")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to CUI embedding pickle")
    parser.add_argument("--graph", type=str, default=None, help="Path to SNOMED graph pickle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load resources ──
    graph_path = args.graph or _default_path("SNOMED_CUI_MAJID_Graph_wSelf.pkl")
    print(f"  {C.DIM}Loading graph: {graph_path}{C.RESET}")
    graph = KnowledgeGraph(graph_path)

    # ── Phase 1 (or direct CUIs) ──
    if args.cuis:
        seed_cuis = args.cuis
        print(f"\n  {C.BOLD}═══ Phase 1: Skipped (direct CUIs) ═══{C.RESET}")
        for cui in seed_cuis:
            in_graph = graph.has_node(cui)
            status = f"{C.GREEN}✓{C.RESET}" if in_graph else f"{C.RED}✗ not in graph{C.RESET}"
            print(f"  {status}  {C.BOLD}{cui}{C.RESET}")
        print()
        get_name = lambda cui: cui  # no name resolution without embeddings
    else:
        emb_path = args.embeddings or _default_path("GraphModel_SNOMED_CUI_Embedding.pkl")
        print(f"  {C.DIM}Loading embeddings: {emb_path}{C.RESET}")
        t0 = time.time()
        index = EmbeddingIndex(emb_path)
        print(f"  {C.DIM}Ready in {time.time() - t0:.1f}s{C.RESET}")

        extractor = TermExtractor()
        phase1_results = find_cuis(args.query, index, top_k=args.top_k, threshold=args.threshold, extractor=extractor)
        print_phase1(args.query, phase1_results)

        seed_cuis = [r["cui"] for r in phase1_results]
        get_name = index.get_name

    if not seed_cuis:
        print(f"  {C.RED}No seed CUIs — nothing to expand.{C.RESET}")
        return

    # ── Phase 2 ──
    allowed = None if args.all_relations else DIAGNOSTIC_RELATIONS

    t0 = time.time()
    expanded = expand_cuis(
        seed_cuis,
        graph,
        depth=args.depth,
        allowed_relations=allowed,
        max_per_hop=args.max_per_hop,
    )
    elapsed_ms = (time.time() - t0) * 1000

    print_expansion(expanded, get_name, max_display=args.max_display)
    print_summary(expanded, elapsed_ms)


if __name__ == "__main__":
    main()
