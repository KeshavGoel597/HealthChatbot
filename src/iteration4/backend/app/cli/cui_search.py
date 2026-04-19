#!/usr/bin/env python3
"""
CLI for testing CUI semantic search.

Usage:
    # Single query
    python -m app.cli.cui_search "my head hurts"

    # Term extraction only (no CUI search)
    python -m app.cli.cui_search "show my full medication history" --terms-only

    # Custom top-k and threshold
    python -m app.cli.cui_search "chest pain" --top-k 5 --threshold 0.8

    # Interactive mode (keep typing queries)
    python -m app.cli.cui_search --interactive

    # Custom embedding path
    python -m app.cli.cui_search "diabetes" --embeddings /path/to/embeddings.pkl
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.cui_search import find_cuis
from app.services.rag.term_extractor import TermExtractor


def print_results(query: str, results: list[dict]) -> None:
    print(f'\n  Query: "{query}"')
    print(f"  {'Rank':<5} {'CUI':<12} {'Score':<8} {'Name'}")
    print(f"  {'─'*5} {'─'*12} {'─'*8} {'─'*40}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<5} {r['cui']:<12} {r['score']:<8.4f} {r['name']}")
    if not results:
        print("  (no results above threshold)")
    print()


def default_embedding_path() -> str:
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(backend_dir, "..", "GraphModel_SNOMED_CUI_Embedding.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUI Semantic Search — map natural language to UMLS concepts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "my head hurts"
  %(prog)s "blood sugar" --top-k 5 --threshold 0.8
  %(prog)s --interactive
        """,
    )
    parser.add_argument("query", nargs="?", default=None, help="Query text (e.g. 'my head hurts')")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode: keep entering queries")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("-t", "--threshold", type=float, default=0.7, help="Min cosine similarity (default: 0.7)")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to CUI embedding pickle")
    parser.add_argument("--vocab", type=str, default=None, help="Path to CUI_Vocab.json")
    parser.add_argument("--show-terms", action="store_true", help="Print extracted medical terms before CUI results")
    parser.add_argument("--terms-only", action="store_true", help="Only run term extraction, skip CUI search")
    return parser.parse_args()


def load_index(args: argparse.Namespace) -> EmbeddingIndex:
    emb_path = args.embeddings or default_embedding_path()
    print(f"Loading index from: {emb_path}")
    t0 = time.time()
    index = EmbeddingIndex(emb_path, vocab_path=args.vocab)
    print(f"Ready in {time.time() - t0:.1f}s\n")
    return index


def run_query(query: str, index: EmbeddingIndex, extractor: TermExtractor, top_k: int, threshold: float, show_terms: bool = False) -> None:
    t0 = time.time()
    extracted = extractor.extract(query)
    results = find_cuis(extracted.terms, index, top_k=top_k, threshold=threshold)
    elapsed_ms = (time.time() - t0) * 1000
    if show_terms:
        print(f"\n  Intent: {extracted.intent}")
        print(f"  Categories: {extracted.categories}")
        print(f"  Terms: {extracted.terms}")
    print_results(query, results)
    print(f"  ({elapsed_ms:.0f}ms, {len(results)} results above {threshold} threshold)\n")


def run_terms_only(query: str, extractor: TermExtractor) -> None:
    t0 = time.time()
    extracted = extractor.extract(query)
    elapsed_ms = (time.time() - t0) * 1000
    print(f'\n  Query: "{query}"')
    print(f"  Intent: {extracted.intent}")
    print(f"  Categories: {extracted.categories}")
    print(f"  Terms: {extracted.terms}")
    print(f"  ({elapsed_ms:.0f}ms, terms-only mode)\n")


def _repl_loop(header: str, worker) -> None:
    print(f"{header} — type a query and press Enter. 'quit' or Ctrl-C to exit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        worker(query)


def interactive_loop(index: EmbeddingIndex, extractor: TermExtractor, top_k: int, threshold: float, show_terms: bool = False) -> None:
    _repl_loop("Interactive mode", lambda q: run_query(q, index, extractor, top_k, threshold, show_terms))


def interactive_terms_loop(extractor: TermExtractor) -> None:
    _repl_loop("Interactive terms-only mode", lambda q: run_terms_only(q, extractor))


def main() -> None:
    args = parse_args()

    if not args.query and not args.interactive:
        print("Error: provide a query or use --interactive mode.\n")
        parse_args()
        sys.exit(1)

    extractor = TermExtractor()

    if args.terms_only:
        if args.interactive:
            interactive_terms_loop(extractor)
        else:
            run_terms_only(args.query, extractor)
        return

    index = load_index(args)

    if args.interactive:
        interactive_loop(index, extractor, args.top_k, args.threshold, args.show_terms)
    else:
        run_query(args.query, index, extractor, args.top_k, args.threshold, args.show_terms)


if __name__ == "__main__":
    main()
