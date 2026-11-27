import math

import pytest

from ranker.metrics import dcg, mrr_at_k, ndcg_at_k, precision_at_k
from ranker.text import features_to_text, normalize_token, tokenize


def test_tokenize_lowercases_strips_accents_and_folds_plurals():
    assert tokenize("Accent Chairs & Wall Décor") == ["accent", "chair", "wall", "decor"]


@pytest.mark.parametrize("word,expected", [
    ("benches", "bench"), ("glasses", "glass"), ("lamps", "lamp"),
    ("accessories", "accessory"), ("glass", "glass"), ("bus", "bus"),
])
def test_normalize_token(word, expected):
    assert normalize_token(word) == expected


def test_features_to_text_keeps_only_textual_values():
    raw = "color : burnt orange|overallwidth:64.7|upholstered : yes|woodspecies : rubberwood"
    assert features_to_text(raw) == "burnt orange rubberwood"


def test_dcg_known_value():
    # (2^2-1)/log2(2) + (2^1-1)/log2(3)
    assert dcg([2, 1]) == pytest.approx(3 + 1 / math.log2(3))


def test_perfect_ranking_scores_one():
    assert ndcg_at_k([2, 1, 0], [0, 1, 2], k=3) == pytest.approx(1.0)


def test_ndcg_is_nan_without_relevant_items():
    assert math.isnan(ndcg_at_k([0, 0], [0, 0]))


def test_worse_ranking_scores_lower():
    assert ndcg_at_k([0, 1, 2], [0, 1, 2], k=3) < ndcg_at_k([2, 1, 0], [0, 1, 2], k=3)


def test_mrr_and_precision():
    assert mrr_at_k([0, 1, 2, 0], k=10) == pytest.approx(1 / 3)
    assert mrr_at_k([1, 1], k=10) == 0.0
    assert precision_at_k([2, 0, 2], k=10) == pytest.approx(0.2)
