import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from ranker.bm25 import BM25Index
from ranker.data import prepare_products
from ranker.engine import FEATURE_NAMES, SearchEngine


@pytest.fixture(scope="module")
def engine():
    products = prepare_products(pd.DataFrame({
        "product_id": range(6),
        "product_name": ["red velvet sofa", "grey velvet sofa", "oak dining table",
                         "round oak dining table", "velvet accent chair", "leather accent chair"],
        "product_class": ["Sofas", "Sofas", "Dining Tables", "Dining Tables",
                          "Accent Chairs", "Accent Chairs"],
        "category_hierarchy": ["Furniture / Living Room"] * 2 + ["Furniture / Dining"] * 2
                              + ["Furniture / Living Room"] * 2,
        "product_features": ["color : red", "color : grey", "woodspecies : oak",
                             "woodspecies : oak", "color : green", "material : leather"],
        "review_count": [10, 0, 5, np.nan, 3, 8],
        "average_rating": [4.5, np.nan, 4.0, np.nan, 3.5, 4.8],
    }))
    return SearchEngine().fit(products)


def test_bm25_prefers_matching_document():
    index = BM25Index().fit(["red velvet sofa", "oak dining table", "velvet accent chair"])
    scores = index.score("velvet sofas")  # plural folds to "sofa"
    assert scores.argmax() == 0
    assert scores[1] == 0


def test_bm25_rare_term_outweighs_common_term():
    index = BM25Index().fit(["velvet sofa", "velvet chair", "velvet bench", "linen sofa"])
    assert index.score("linen")[3] > index.score("velvet")[0]


def test_unknown_query_scores_zero():
    index = BM25Index().fit(["velvet sofa"])
    assert index.score("zzz").sum() == 0


def test_empty_field_does_not_crash():
    index = BM25Index().fit(["", ""])
    assert index.score("sofa").tolist() == [0.0, 0.0]


def test_stage1_search_finds_sofa(engine):
    pids, _ = engine.search("red velvet sofa", k=3)
    assert pids[0] == 0


def test_intent_classifier_predicts_class(engine):
    assert engine.intent.top_k("oak table", 1)[0][0] == "Dining Tables"


def test_pair_features_shape_and_values(engine):
    signals = engine.query_signals("velvet sofa")
    X = engine.pair_features(signals, [0, 2])
    assert X.shape == (2, len(FEATURE_NAMES))
    col = FEATURE_NAMES.index("name_coverage")
    assert X[0, col] == 1.0 and X[1, col] == 0.0


def test_reranking_with_a_model(engine):
    signals = engine.query_signals("velvet")
    X = engine.pair_features(signals, range(6))
    y = np.array([2, 2, 0, 0, 1, 0])
    model = HistGradientBoostingRegressor(max_iter=20, min_samples_leaf=1).fit(X, y)
    pids, scores = engine.search("velvet", model=model, k=2)
    assert len(pids) == 2 and scores[0] >= scores[1]
