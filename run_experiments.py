#!/usr/bin/env python3
"""Reproduce every number in the README.

    python run_experiments.py --data data/WANDS/dataset

Steps
  1. Index the catalogue (5 BM25 fields + character n-grams) and train the
     query-intent classifier on product titles.
  2. For every query, build features for its judged products and for its
     stage-1 candidates (top-100 multi-field BM25).
  3. 5-fold GroupKFold over *queries*: train a gradient-boosted classifier
     (Irrelevant / Partial / Exact) on 4 folds, rank the unseen 5th fold's
     candidates by expected gain 3*P(Exact) + P(Partial).
  4. Write results/metrics.json, results/results.md and the final model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from ranker.data import load_wands  # noqa: E402
from ranker.engine import FEATURE_NAMES, INTENT_FEATURES, SearchEngine  # noqa: E402
from ranker.metrics import mrr_at_k, ndcg_at_k, precision_at_k  # noqa: E402

FULL = list(FEATURE_NAMES)
NO_INTENT = [f for f in FEATURE_NAMES if f not in INTENT_FEATURES]


GAINS = np.array([0.0, 1.0, 3.0])  # NDCG gain of Irrelevant / Partial / Exact


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, random_state=42,
    )


def expected_gain(model: HistGradientBoostingClassifier, X: np.ndarray) -> np.ndarray:
    """Rank by expected NDCG gain: 3 * P(Exact) + 1 * P(Partial)."""
    return model.predict_proba(X) @ GAINS[model.classes_.astype(int)]


def training_rows(pairs: pd.DataFrame, queries: set, with_negatives: bool) -> pd.DataFrame:
    """Judged pairs, optionally plus unjudged stage-1 candidates labelled as irrelevant."""
    keep = pairs["query_id"].isin(queries) & (pairs["rel"].notna() | (with_negatives & pairs["is_candidate"]))
    rows = pairs[keep].copy()
    rows["rel"] = rows["rel"].fillna(0).astype(int)
    return rows


def ranking_metrics(ranked_pids, rel_map, judged_rels) -> dict:
    ranked_rels = [rel_map.get(int(p), 0) for p in ranked_pids]  # unjudged counts as irrelevant
    return {
        "ndcg@10": ndcg_at_k(ranked_rels, judged_rels, 10),
        "mrr@10": mrr_at_k(ranked_rels, 10) if (judged_rels == 2).any() else np.nan,
        "exact_p@10": precision_at_k(ranked_rels, 10),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/WANDS/dataset", help="folder with product.csv, query.csv, label.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=100)
    args = ap.parse_args()
    out, artifacts = Path(args.out), Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    wands = load_wands(args.data)
    engine = SearchEngine().fit(wands.products)
    print(f"[1/4] indexed {len(wands.products):,} products in {time.time() - t0:.0f}s")

    # Query-intent accuracy on real queries (the classifier never saw them).
    q = wands.queries
    known = q["query_class"].isin(engine.intent.class_index)
    proba = engine.intent.predict_proba(q.loc[known, "query"])
    y_true = q.loc[known, "query_class"].map(engine.intent.class_index).to_numpy()
    top3 = np.argsort(-proba, axis=1)[:, :3]
    intent = {
        "eval_queries": int(known.sum()),
        "n_classes": int(len(engine.intent.classes_)),
        "top1_accuracy": float(np.mean(top3[:, 0] == y_true)),
        "top3_accuracy": float(np.mean([y_true[i] in top3[i] for i in range(len(y_true))])),
    }
    print(f"      query-intent top-1 {intent['top1_accuracy']:.3f} | top-3 {intent['top3_accuracy']:.3f}")

    # ---------------------------------------------------------------- pairs
    labels_by_query = {qid: g for qid, g in wands.labels.groupby("query_id")}
    tables, per_query, info = [], [], {}
    for qid, query in zip(q["query_id"], q["query"]):
        sig = engine.query_signals(query)
        judged = labels_by_query.get(qid)
        rel_map = ({int(p): int(r) for p, r in zip(judged["product_id"], judged["rel"])}
                   if judged is not None else {})
        judged_rels = np.array(list(rel_map.values()), dtype=int)
        cand = engine.retrieve(sig, args.candidates)

        pids = np.array(sorted(set(rel_map) | set(cand.tolist())), dtype=int)
        df = pd.DataFrame(engine.pair_features(sig, pids), columns=FEATURE_NAMES)
        df.insert(0, "query_id", qid)
        df.insert(1, "product_id", pids)
        df["rel"] = [rel_map.get(int(p), np.nan) for p in pids]
        df["is_candidate"] = np.isin(pids, cand)
        tables.append(df)
        info[qid] = {"rel_map": rel_map, "judged_rels": judged_rels, "cand": cand}

        # Unsupervised baselines over the full catalogue.
        top_all = np.argsort(-sig["bm25"]["all"], kind="stable")[:10]
        per_query.append({"query_id": qid, "method": "bm25_single_field",
                          **ranking_metrics(top_all, rel_map, judged_rels)})
        per_query.append({"query_id": qid, "method": "bm25_multi_field",
                          **ranking_metrics(cand[:10], rel_map, judged_rels)})
    pairs = pd.concat(tables, ignore_index=True)
    judged_pairs = int(pairs["rel"].notna().sum())
    print(f"[2/4] built {len(pairs):,} (query, product) feature rows ({judged_pairs:,} judged)")

    # --------------------------------------------------- cross-validation
    query_ids = np.array(sorted(pairs["query_id"].unique()))
    gkf = GroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    variants = [  # (name, feature columns, add unjudged candidates as negatives)
        ("ltr_judged_only", FULL, False),
        ("ltr_no_intent", NO_INTENT, True),
        ("ltr_with_intent", FULL, True),
    ]
    pool_rows, importances = [], []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(query_ids, groups=query_ids), start=1):
        train_q, test_q = set(query_ids[tr_idx]), set(query_ids[te_idx])
        test = pairs[pairs["query_id"].isin(test_q)]

        for method, cols, negatives in variants:
            train = training_rows(pairs, train_q, negatives)
            model = make_model().fit(train[cols].to_numpy(), train["rel"].to_numpy())
            scored = test.assign(score=expected_gain(model, test[cols].to_numpy()))
            for qid, g in scored.groupby("query_id"):
                meta = info[qid]
                cand_rows = g[g["is_candidate"]].sort_values("score", ascending=False, kind="stable")
                per_query.append({"query_id": qid, "method": method,
                                  **ranking_metrics(cand_rows["product_id"].to_numpy()[:10],
                                                    meta["rel_map"], meta["judged_rels"])})
                # Judged-pool setting: re-rank only the products that were labelled.
                pool = g[g["rel"].notna()]
                rankers = [(method, "score")]
                if method == variants[0][0]:  # BM25 rankings need no model; record them once
                    rankers += [("bm25_single_field", "bm25_all"), ("bm25_multi_field", "stage1_score")]
                for name, col in rankers:
                    ranked = pool.sort_values(col, ascending=False, kind="stable")["rel"].to_numpy()
                    pool_rows.append({"query_id": qid, "method": name,
                                      "pool_ndcg@10": ndcg_at_k(ranked, pool["rel"].to_numpy(), 10)})
            if method == "ltr_with_intent" and fold == 1:
                held = test[test["rel"].notna()]
                imp = permutation_importance(model, held[cols].to_numpy(), held["rel"].astype(int).to_numpy(),
                                             scoring="neg_log_loss", n_repeats=2, random_state=0, n_jobs=1)
                importances.append(imp.importances_mean)
        print(f"      fold {fold}/{args.folds} done ({len(train):,} training rows)")
    print("[3/4] cross-validation finished")

    # ---------------------------------------------------------- reporting
    pq = pd.DataFrame(per_query)
    pool_df = pd.DataFrame(pool_rows)
    order = ["bm25_single_field", "bm25_multi_field", "ltr_judged_only", "ltr_no_intent", "ltr_with_intent"]
    labels = {
        "bm25_single_field": "BM25, one field (all text)",
        "bm25_multi_field": "Multi-field BM25 (stage 1)",
        "ltr_judged_only": "+ GBDT re-ranker, judged pairs only",
        "ltr_no_intent": "+ GBDT re-ranker + retrieved negatives, no intent features",
        "ltr_with_intent": "+ GBDT re-ranker + retrieved negatives + query intent (final)",
    }
    summary = pq.groupby("method")[["ndcg@10", "mrr@10", "exact_p@10"]].mean().reindex(order)
    pool_summary = pool_df.groupby("method")["pool_ndcg@10"].mean().reindex(order)

    wide = pq.pivot(index="query_id", columns="method", values="ndcg@10").dropna()
    better = int((wide["ltr_with_intent"] > wide["bm25_multi_field"] + 1e-9).sum())
    worse = int((wide["ltr_with_intent"] < wide["bm25_multi_field"] - 1e-9).sum())
    base, final = summary.loc["bm25_single_field", "ndcg@10"], summary.loc["ltr_with_intent", "ndcg@10"]

    imp = pd.Series(np.mean(importances, axis=0), index=FULL).sort_values(ascending=False)
    metrics = {
        "dataset": {"products": len(wands.products), "queries": len(q), "judgements": len(wands.labels)},
        "query_intent": intent,
        "full_catalogue": {m: summary.loc[m].round(4).to_dict() for m in order},
        "judged_pool_ndcg@10": pool_summary.round(4).to_dict(),
        "ndcg@10_queries_with_relevant_items": int(pq[pq.method == "ltr_with_intent"]["ndcg@10"].notna().sum()),
        "final_vs_stage1_queries": {"better": better, "worse": worse, "tied": int(len(wide) - better - worse)},
        "relative_ndcg@10_gain_vs_single_field_bm25": round(float(final / base - 1), 4),
        "permutation_importance_logloss_increase": imp.round(4).to_dict(),
        "settings": {"folds": args.folds, "candidates": args.candidates, "cv": "GroupKFold by query"},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        "# Results",
        "",
        f"{args.folds}-fold cross-validation, split by query (each query is scored by a model that never saw it).",
        "Full-catalogue setting: rank all 42,994 products; products nobody judged count as irrelevant.",
        "",
        "| Method | NDCG@10 | MRR@10 (Exact) | Exact P@10 | NDCG@10 (judged pool) |",
        "|---|---|---|---|---|",
    ]
    for m in order:
        r = summary.loc[m]
        lines.append(f"| {labels[m]} | {r['ndcg@10']:.3f} | {r['mrr@10']:.3f} | {r['exact_p@10']:.3f} "
                     f"| {pool_summary.loc[m]:.3f} |")
    lines += [
        "",
        f"Final model vs. multi-field BM25, per query: better on {better}, worse on {worse}, "
        f"tied on {len(wide) - better - worse}.",
        "",
        f"Query-intent classifier (trained on product titles only): top-1 accuracy "
        f"{intent['top1_accuracy']:.1%}, top-3 accuracy {intent['top3_accuracy']:.1%} on "
        f"{intent['eval_queries']} real queries across {intent['n_classes']} product classes.",
        "",
        "## Most useful features (permutation importance on fold 1, increase in log-loss)",
        "",
        "| Feature | Importance |",
        "|---|---|",
        *[f"| `{name}` | {value:.4f} |" for name, value in imp.head(10).items()],
        "",
    ]
    (out / "results.md").write_text("\n".join(lines))

    # --------------------------------------------- final model for serving
    all_rows = training_rows(pairs, set(query_ids), with_negatives=True)
    final_model = make_model().fit(all_rows[FULL].to_numpy(), all_rows["rel"].to_numpy())
    joblib.dump({"model": final_model, "features": FULL}, artifacts / "ltr_model.joblib")
    print(f"[4/4] wrote {out}/results.md, {out}/metrics.json and {artifacts}/ltr_model.joblib "
          f"in {time.time() - t0:.0f}s total\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
