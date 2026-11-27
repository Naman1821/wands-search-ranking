#!/usr/bin/env python3
"""Search the WANDS catalogue with the trained two-stage ranker.

    python search.py "modern velvet sofa" "outdoor dining set for 6"
    python search.py                 # interactive prompt
    python search.py --bm25-only "bar stool with backrest"

Run `python run_experiments.py` first: it trains and saves the re-ranker.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402

from ranker.data import load_wands  # noqa: E402
from ranker.engine import FEATURE_NAMES, SearchEngine  # noqa: E402


def show(engine: SearchEngine, query: str, bundle: dict | None, k: int) -> None:
    intents = " > ".join(c for c, _ in engine.intent.top_k(query, 3))
    print(f'\nQuery: "{query}"\n  predicted intent: {intents}')
    model = bundle["model"] if bundle else None
    feature_idx = [FEATURE_NAMES.index(f) for f in bundle["features"]] if bundle else None
    pids, scores = engine.search(query, model=model, k=k, feature_idx=feature_idx)
    if len(pids) == 0:
        print("  no product matches any query term")
    for rank, (pid, score) in enumerate(zip(pids, scores), start=1):
        row = engine.products.iloc[int(pid)]
        cls = row["product_class"] or "-"
        print(f"  {rank:>2}. {row['product_name'][:60]:<60}  [{cls[:28]}]  {score:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="*", help="queries to run; omit for an interactive prompt")
    ap.add_argument("--data", default="data/WANDS/dataset")
    ap.add_argument("--model", default="artifacts/ltr_model.joblib")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--bm25-only", action="store_true", help="skip the re-ranker (stage 1 only)")
    args = ap.parse_args()

    bundle = None
    if not args.bm25_only:
        path = Path(args.model)
        if not path.exists():
            sys.exit(f"No model at {path}. Run `python run_experiments.py` first, or pass --bm25-only.")
        bundle = joblib.load(path)

    started = time.time()
    engine = SearchEngine().fit(load_wands(args.data).products)
    print(f"Indexed {engine.n_products:,} products in {time.time() - started:.0f}s "
          f"({'BM25 only' if bundle is None else 'BM25 + re-ranker'})")

    if args.queries:
        for query in args.queries:
            show(engine, query, bundle, args.k)
        return
    while True:
        try:
            query = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        show(engine, query, bundle, args.k)


if __name__ == "__main__":
    main()
