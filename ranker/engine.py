"""Two-stage product search: multi-field BM25 retrieval, then learned re-ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .bm25 import BM25Index
from .intent import QueryIntentClassifier
from .text import strip_accents, tokenize

FIELDS = ("all", "name", "classcat", "features", "desc")

# Stage-1 field weights, fixed up front (not tuned on the evaluation queries).
STAGE1_WEIGHTS = {"name": 3.0, "classcat": 2.0, "features": 1.0, "desc": 0.5}

FEATURE_NAMES = [
    # Lexical match, both absolute and relative to the query's best-matching product.
    "bm25_all", "bm25_name", "bm25_classcat", "bm25_features", "bm25_desc",
    "bm25_all_rel", "bm25_name_rel", "bm25_classcat_rel", "bm25_features_rel", "bm25_desc_rel",
    "bm25_all_log_rank", "bm25_name_log_rank",
    "stage1_score", "stage1_rel", "stage1_log_rank",
    "char_cosine_name", "char_cosine_rel",
    "name_coverage", "text_coverage", "name_phrase_match",
    # Query understanding.
    "intent_prob", "intent_prob_rel", "intent_rank", "intent_top1_match",
    # Pseudo-relevance feedback: does the product's class dominate the top results?
    "prf_class_share",
    # Product priors and query shape.
    "log_review_count", "average_rating", "query_len",
]
INTENT_FEATURES = ["intent_prob", "intent_prob_rel", "intent_rank", "intent_top1_match"]
PRF_DEPTH = 20  # how many stage-1 results vote on the query's likely classes
_COL = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _rank(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(order, 1-based rank of every item); items scoring 0 share the last rank."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float32)
    ranks[order] = np.arange(1, scores.size + 1)
    ranks[scores <= 0] = scores.size
    return order, ranks


def _safe_div(values: np.ndarray, denom: float) -> np.ndarray:
    return values / denom if denom > 0 else np.zeros_like(values)


def _char_preprocess(text: str) -> str:
    return strip_accents(text.lower())


class SearchEngine:
    def __init__(self, stage1_weights: dict | None = None, k1: float = 1.2, b: float = 0.75):
        self.stage1_weights = dict(stage1_weights or STAGE1_WEIGHTS)
        self.k1, self.b = k1, b

    # ------------------------------------------------------------------ build
    def fit(self, products: pd.DataFrame) -> "SearchEngine":
        self.products = products
        self.n_products = len(products)
        self.indexes = {f: BM25Index(self.k1, self.b).fit(products[f"field_{f}"]) for f in FIELDS}

        # Character n-grams catch typos and partial words ("recliner" vs "reclining").
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, preprocessor=_char_preprocess,
        )
        self.name_char = self.char_vectorizer.fit_transform(products["field_name"])

        self.intent = QueryIntentClassifier().fit(products["field_name"], products["classes"])

        # Per-product lookups used to build (query, product) features.
        name_tokens = [tokenize(t) for t in products["field_name"]]
        self.name_token_sets = [set(t) for t in name_tokens]
        self.name_joined = [" " + " ".join(t) + " " for t in name_tokens]
        self.text_token_sets = [
            names | set(tokenize(classcat)) | set(tokenize(feats))
            for names, classcat, feats in zip(
                self.name_token_sets, products["field_classcat"], products["field_features"]
            )
        ]
        self.product_class_ids = [
            [self.intent.class_index[c] for c in classes if c in self.intent.class_index]
            for classes in products["classes"]
        ]
        self.log_reviews = np.log1p(products["review_count"].to_numpy(dtype=float))
        self.avg_rating = products["average_rating"].to_numpy(dtype=float)
        return self

    # ---------------------------------------------------- per-query signals
    def query_signals(self, query: str) -> dict:
        """Scores of the query against the whole catalogue, computed once per query."""
        bm25 = {f: index.score(query) for f, index in self.indexes.items()}
        stage1 = sum(w * bm25[f] for f, w in self.stage1_weights.items())
        order, ranks = _rank(stage1)
        field_ranks = {f: _rank(bm25[f])[1] for f in ("all", "name")}

        query_char = self.char_vectorizer.transform([query])
        char = np.asarray((self.name_char @ query_char.T).todense()).ravel()

        proba = self.intent.predict_proba([query])[0]
        class_rank = np.empty_like(proba)
        class_rank[np.argsort(-proba)] = np.arange(1, proba.size + 1)

        # Pseudo-relevance feedback: share of the top stage-1 results in each class.
        top = order[:PRF_DEPTH]
        top = top[stage1[top] > 0]
        prf_share = np.zeros(proba.size, dtype=np.float32)
        for pid in top:
            for c in set(self.product_class_ids[pid]):
                prf_share[c] += 1.0
        if top.size:
            prf_share /= top.size

        return {
            "query": query, "bm25": bm25, "bm25_max": {f: float(s.max()) for f, s in bm25.items()},
            "stage1": stage1, "stage1_max": float(stage1.max()), "order": order, "ranks": ranks,
            "field_ranks": field_ranks, "char": char, "char_max": float(char.max()),
            "intent_proba": proba, "intent_class_rank": class_rank, "top_class": int(np.argmax(proba)),
            "top_prob": float(proba.max()), "prf_share": prf_share, "q_tokens": tokenize(query),
        }

    def retrieve(self, signals: dict, k: int = 100) -> np.ndarray:
        """Stage 1: top-k products by multi-field BM25 (only products matching a query term)."""
        top = signals["order"][:k]
        return top[signals["stage1"][top] > 0]

    def pair_features(self, signals: dict, product_ids) -> np.ndarray:
        """Feature matrix for (query, product) pairs; NaN where a feature is undefined."""
        pids = np.asarray(product_ids, dtype=int)
        X = np.full((pids.size, len(FEATURE_NAMES)), np.nan, dtype=np.float32)

        for f in FIELDS:
            X[:, _COL[f"bm25_{f}"]] = signals["bm25"][f][pids]
            X[:, _COL[f"bm25_{f}_rel"]] = _safe_div(signals["bm25"][f][pids], signals["bm25_max"][f])
        for f in ("all", "name"):
            X[:, _COL[f"bm25_{f}_log_rank"]] = np.log1p(signals["field_ranks"][f][pids])
        X[:, _COL["stage1_score"]] = signals["stage1"][pids]
        X[:, _COL["stage1_rel"]] = _safe_div(signals["stage1"][pids], signals["stage1_max"])
        X[:, _COL["stage1_log_rank"]] = np.log1p(signals["ranks"][pids])
        X[:, _COL["char_cosine_name"]] = signals["char"][pids]
        X[:, _COL["char_cosine_rel"]] = _safe_div(signals["char"][pids], signals["char_max"])

        q_tokens = signals["q_tokens"]
        q_set = set(q_tokens)
        q_phrase = " " + " ".join(q_tokens) + " "
        proba = signals["intent_proba"]
        class_rank = signals["intent_class_rank"]
        top_class, top_prob = signals["top_class"], signals["top_prob"]
        prf_share = signals["prf_share"]
        for row, pid in enumerate(pids):
            if q_set:
                X[row, _COL["name_coverage"]] = len(q_set & self.name_token_sets[pid]) / len(q_set)
                X[row, _COL["text_coverage"]] = len(q_set & self.text_token_sets[pid]) / len(q_set)
                X[row, _COL["name_phrase_match"]] = float(q_phrase in self.name_joined[pid])
            classes = self.product_class_ids[pid]
            if classes:
                best_prob = max(proba[c] for c in classes)
                X[row, _COL["intent_prob"]] = best_prob
                X[row, _COL["intent_prob_rel"]] = best_prob / top_prob if top_prob > 0 else 0.0
                X[row, _COL["intent_rank"]] = min(class_rank[c] for c in classes)
                X[row, _COL["intent_top1_match"]] = float(top_class in classes)
                X[row, _COL["prf_class_share"]] = max(prf_share[c] for c in classes)

        X[:, _COL["log_review_count"]] = self.log_reviews[pids]
        X[:, _COL["average_rating"]] = self.avg_rating[pids]
        X[:, _COL["query_len"]] = len(q_tokens)
        return X

    # ------------------------------------------------------------- serving
    def search(self, query: str, model=None, k: int = 10, candidates: int = 100,
               feature_idx=None) -> tuple[np.ndarray, np.ndarray]:
        """Return (product_ids, scores) for the top-k results.

        Without a model this is plain multi-field BM25; with a model the
        stage-1 candidates are re-ranked by the model's predicted relevance.
        """
        signals = self.query_signals(query)
        cand = self.retrieve(signals, candidates)
        if model is None or cand.size == 0:
            return cand[:k], signals["stage1"][cand[:k]]
        X = self.pair_features(signals, cand)
        if feature_idx is not None:
            X = X[:, feature_idx]
        if hasattr(model, "predict_proba"):  # classifier: rank by expected gain 3*P(Exact) + P(Partial)
            gains = np.array([0.0, 1.0, 3.0])[model.classes_.astype(int)]
            scores = model.predict_proba(X) @ gains
        else:
            scores = model.predict(X)
        best = np.argsort(-scores, kind="stable")[:k]
        return cand[best], scores[best]
