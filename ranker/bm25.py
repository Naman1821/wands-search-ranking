"""Sparse BM25 built on scikit-learn's CountVectorizer.

Every (document, term) contribution is pre-computed once at index time, so
scoring a query is just a sum over the query's term columns.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1  # term-frequency saturation
        self.b = b    # document-length normalisation

    def fit(self, documents) -> "BM25Index":
        documents = list(documents)
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        try:
            tf = self.vectorizer.fit_transform(documents).tocsr().astype(np.float32)
        except ValueError as err:  # every document is empty, e.g. a field nobody filled in
            if "empty vocabulary" not in str(err):
                raise
            self.vectorizer.vocabulary_ = {}
            tf = sp.csr_matrix((len(documents), 0), dtype=np.float32)
        n_docs = tf.shape[0]

        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean()) or 1.0
        row_of_nnz = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        length_norm = self.k1 * (1.0 - self.b + self.b * doc_len[row_of_nnz] / avgdl)
        weights = tf.data * (self.k1 + 1.0) / (tf.data + length_norm) * self.idf[tf.indices]

        self.weights = sp.csr_matrix(
            (weights.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape
        ).tocsc()  # column slicing is what scoring needs
        self.vocab = self.vectorizer.vocabulary_
        return self

    def term_ids(self, query: str) -> list[int]:
        return sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query` (zeros if no term matches)."""
        ids = self.term_ids(query)
        if not ids:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, ids].sum(axis=1)).ravel()
