#!/usr/bin/env python3
"""
CLI wrapper for the Graph-Augmented Semantic RAG pipeline.

Usage:
    python cli.py --query "kidney function results" --emr-path patient_data.json

Optional flags:
    --cui-emb-path   Path to CUI embedding pickle  (default: GraphModel_SNOMED_CUI_Embedding.pkl)
    --graph-path     Path to SNOMED graph pickle    (default: SNOMED_CUI_MAJID_Graph_wSelf.pkl)
    --cui-name-path  Path to CUI name pickle        (default: sm_t047_cui_aui_eng.pkl)
    --date           Reference date (DD-Mon-YYYY)   (default: today)
    --json-out       Path to write JSON output       (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from rag import run_pipeline


def _resolve(path: str) -> str:
    """Resolve a path relative to the script directory if not absolute."""
    if os.path.isabs(path):
        return path
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph-Augmented Semantic RAG for EMR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Medical query string.",
    )
    parser.add_argument(
        "--emr-path",
        default="patient_data.json",
        help="Path to Ruby-style EMR JSON file.",
    )
    parser.add_argument(
        "--cui-emb-path",
        default="GraphModel_SNOMED_CUI_Embedding.pkl",
        help="Path to CUI embedding pickle.",
    )
    parser.add_argument(
        "--graph-path",
        default="SNOMED_CUI_MAJID_Graph_wSelf.pkl",
        help="Path to SNOMED knowledge graph pickle.",
    )
    parser.add_argument(
        "--cui-name-path",
        default="sm_t047_cui_aui_eng.pkl",
        help="Path to CUI name/semantic-type pickle.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Reference date in DD-Mon-YYYY format (e.g. 09-Mar-2026). Default: today.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the full JSON output.",
    )

    args = parser.parse_args()

    # Resolve paths
    emr_path = _resolve(args.emr_path)
    cui_emb_path = _resolve(args.cui_emb_path)
    graph_path = _resolve(args.graph_path)
    cui_name_path = _resolve(args.cui_name_path)

    # Parse reference date
    ref_date = None
    if args.date:
        try:
            ref_date = datetime.strptime(args.date, "%d-%b-%Y")
        except ValueError:
            print(f"Error: could not parse date '{args.date}'. Use DD-Mon-YYYY format.", file=sys.stderr)
            sys.exit(1)

    # Validate files exist
    for label, path in [
        ("EMR", emr_path),
        ("CUI embeddings", cui_emb_path),
        ("SNOMED graph", graph_path),
        ("CUI names", cui_name_path),
    ]:
        if not os.path.isfile(path):
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Run pipeline
    result = run_pipeline(
        query=args.query,
        emr_path=emr_path,
        cui_emb_path=cui_emb_path,
        graph_path=graph_path,
        cui_name_path=cui_name_path,
        reference_date=ref_date,
    )

    # Print prompt
    print("\n" + "=" * 72)
    print(result["prompt"])
    print("=" * 72)

    # Print entities
    if result.get("entities"):
        print(f"\nExtracted Entities: {result['entities']}")

    # Print summary
    print(f"\nSeed CUIs ({len(result['seed_cuis'])}):")
    for item in result["seed_cuis"]:
        print(f"  • {item['cui']}  {item['name']}  (sim={item['score']:.3f})")

    print(f"\nReasoning Paths ({len(result['reasoning_paths'])}):")
    for i, p in enumerate(result["reasoning_paths"], 1):
        nodes_str = " → ".join(p["nodes"])
        rels_str = ", ".join(p["relations"])
        print(f"  {i}. {nodes_str}  [{rels_str}]  (score={p['score']:.3f})")

    # Print path-grounded records
    pg_records = result.get("path_grounded_records", [])
    total_grounded = sum(len(pg["grounded_records"]) for pg in pg_records)
    if total_grounded > 0:
        print(f"\nPath-Grounded Patient Records ({total_grounded} total):")
        for pg in pg_records:
            recs = pg.get("grounded_records", [])
            if not recs:
                continue
            path_num = pg["path_index"] + 1
            summary = pg.get("path_summary", "").split("\n")[0]
            print(f"  Path {path_num}: {summary}")
            for r in recs:
                rec = r["record"]
                rtype = rec.get("record_type", "?")
                concept = r["matched_concept"]
                sim = r["similarity"]
                # Compact display
                display_fields = {k: v for k, v in rec.items() if k != "record_type"}
                print(f"    [{rtype}] → concept: {concept} (sim={sim:.3f})")
                print(f"      {json.dumps(display_fields, ensure_ascii=False)}")

    print(f"\nEvidence ({len(result['evidence'])}):")
    for i, ev in enumerate(result["evidence"], 1):
        print(f"  {i}. [{ev['type']}] {ev['text']}  (score={ev['score']:.3f})")

    print(f"\nElapsed: {result['elapsed_seconds']:.2f}s")

    # Write JSON if requested
    if args.json_out:
        out_path = _resolve(args.json_out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nJSON output written to: {out_path}")


if __name__ == "__main__":
    main()
