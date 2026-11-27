"""Ranking metrics. `rels` are graded labels: 2 = Exact, 1 = Partial, 0 = Irrelevant/unjudged."""
from __future__ import annotations

import numpy as np


def dcg(rels) -> float:
    rels = np.asarray(rels, dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rels.size + 2))
    return float(np.sum((2.0 ** rels - 1.0) / discounts))


def ndcg_at_k(ranked_rels, judged_rels, k: int = 10) -> float:
    """NDCG@k. The ideal ranking uses every judged label for the query.

    Returns NaN when the query has no relevant product at all.
    """
    ideal = np.sort(np.asarray(judged_rels, dtype=float))[::-1][:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return float("nan")
    return dcg(np.asarray(ranked_rels, dtype=float)[:k]) / idcg


def mrr_at_k(ranked_rels, k: int = 10, target: int = 2) -> float:
    """Reciprocal rank of the first result with rel >= target (Exact by default)."""
    for position, rel in enumerate(np.asarray(ranked_rels)[:k], start=1):
        if rel >= target:
            return 1.0 / position
    return 0.0


def precision_at_k(ranked_rels, k: int = 10, target: int = 2) -> float:
    """Share of the top-k slots holding a result with rel >= target."""
    top = np.asarray(ranked_rels)[:k]
    return float(np.sum(top >= target)) / k
