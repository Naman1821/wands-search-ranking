"""Query-intent classifier: predict which product class a search query wants.

It is trained only on catalogue data (product title -> product class), which
is weak supervision: titles are short, query-like texts whose class is
already known. No query labels are used for training, so its accuracy on the
real WANDS queries is an honest held-out number.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

from .text import tokenize


class QueryIntentClassifier:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def fit(self, titles, class_lists) -> "QueryIntentClassifier":
        texts, labels = [], []
        for title, classes in zip(titles, class_lists):
            for cls in classes:  # multi-class products contribute one row per class
                texts.append(title)
                labels.append(cls)
        self.classes_ = np.array(sorted(set(labels)))
        self.class_index = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([self.class_index[c] for c in labels])

        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(texts)
        self.model = ComplementNB(alpha=self.alpha).fit(X, y)
        return self

    def predict_proba(self, queries) -> np.ndarray:
        """(n_queries, n_classes) probabilities, columns ordered like `classes_`."""
        return self.model.predict_proba(self.vectorizer.transform(list(queries)))

    def top_k(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        proba = self.predict_proba([query])[0]
        best = np.argsort(-proba)[:k]
        return [(str(self.classes_[i]), float(proba[i])) for i in best]
