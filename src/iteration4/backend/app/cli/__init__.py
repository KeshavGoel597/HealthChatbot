#!/usr/bin/env python3
"""
CLI for testing CUI semantic search.

Usage:
    # Single query
    python -m app.cli.cui_search "my head hurts"

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

# Add backend/ to path so app.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.cui_search import find_cuis
from app.services.rag.term_extractor import TermExtractor


def print_results(query: str, results: list[dict]) -> None:
    """Pretty-print CUI search results for human verification."""
    print(f'\n  Query: "{query}"')
    print(f"  {'Rank':<5} {'CUI':<12} {'Score':<8} {'Name'}")
    print(f"  {'─'*5} {'─'*12} {'─'*8} {'─'*40}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<5} {r['cui']:<12} {r['score']:<8.4f} {r['name']}")
    if not results:
        print("  (no results above threshold)")
    print()


def default_embedding_path() -> str:
    """Resolve default path: backend/../../GraphModel_SNOMED_CUI_Embedding.pkl"""
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
    return parser.parse_args()


def load_index(args: argparse.Namespace) -> EmbeddingIndex:
    """Load the embedding index, printing timing info."""
    emb_path = args.embeddings or default_embedding_path()
    print(f"Loading index from: {emb_path}")
    t0 = time.time()
    index = EmbeddingIndex(emb_path, vocab_path=args.vocab)
    print(f"Ready in {time.time() - t0:.1f}s\n")
    return index


def run_query(query: str, index: EmbeddingIndex, extractor: TermExtractor, top_k: int, threshold: float) -> None:
    """Run a single query and print results."""
    t0 = time.time()
    results = find_cuis(query, index, top_k=top_k, threshold=threshold, extractor=extractor)
    elapsed_ms = (time.time() - t0) * 1000
    print_results(query, results)
    print(f"  ({elapsed_ms:.0f}ms, {len(results)} results above {threshold} threshold)\n")


def interactive_loop(index: EmbeddingIndex, extractor: TermExtractor, top_k: int, threshold: float) -> None:
    """REPL: keep accepting queries until Ctrl-C or 'quit'."""
    print("Interactive mode — type a query and press Enter. 'quit' or Ctrl-C to exit.\n")
    while True:
        try:
            query = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        run_query(query, index, extractor, top_k, threshold)


def main() -> None:
    args = parse_args()

    if not args.query and not args.interactive:
        print("Error: provide a query or use --interactive mode.\n")
        parse_args()  # will print help
        sys.exit(1)

    index = load_index(args)
    extractor = TermExtractor()

    if args.interactive:
        interactive_loop(index, extractor, args.top_k, args.threshold)
    else:
        run_query(args.query, index, extractor, args.top_k, args.threshold)


if __name__ == "__main__":
    main()
