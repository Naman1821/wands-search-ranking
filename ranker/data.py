"""Load the WANDS dataset into tidy pandas tables."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .text import features_to_text

# Graded relevance used for NDCG (gain = 2^rel - 1, so Exact=3, Partial=1).
LABEL_TO_REL = {"Exact": 2, "Partial": 1, "Irrelevant": 0}


@dataclass
class Wands:
    products: pd.DataFrame  # one row per product; row i has product_id == i
    queries: pd.DataFrame   # query_id, query, query_class
    labels: pd.DataFrame    # query_id, product_id, rel (0 / 1 / 2)


def load_wands(data_dir: str | Path) -> Wands:
    data_dir = Path(data_dir)
    products = pd.read_csv(data_dir / "product.csv", sep="\t")
    queries = pd.read_csv(data_dir / "query.csv", sep="\t")
    labels = pd.read_csv(data_dir / "label.csv", sep="\t")

    products = products.rename(columns={"category hierarchy": "category_hierarchy"})
    products = products.sort_values("product_id").reset_index(drop=True)
    if not (products["product_id"].to_numpy() == np.arange(len(products))).all():
        raise ValueError("Expected product_id to run 0..N-1")
    products = prepare_products(products)

    labels["rel"] = labels["label"].map(LABEL_TO_REL).astype(np.int8)
    labels = labels[["query_id", "product_id", "rel"]].reset_index(drop=True)
    return Wands(products, queries.reset_index(drop=True), labels)


def prepare_products(products: pd.DataFrame) -> pd.DataFrame:
    """Add the searchable text fields the engine indexes."""
    products = products.copy()
    text_cols = ["product_name", "product_class", "category_hierarchy",
                 "product_description", "product_features"]
    for col in text_cols:
        if col not in products:
            products[col] = ""
    for col in ["rating_count", "average_rating", "review_count"]:
        if col not in products:
            products[col] = np.nan
    products[text_cols] = products[text_cols].fillna("")

    # A product can belong to several classes, e.g. "Beds|Kids Beds".
    products["classes"] = products["product_class"].map(
        lambda s: [c.strip() for c in s.split("|") if c.strip()]
    )

    # Searchable fields, indexed separately so the ranker can weigh them.
    products["field_name"] = products["product_name"]
    products["field_classcat"] = (
        products["product_class"].str.replace("|", " ", regex=False)
        + " "
        + products["category_hierarchy"].str.replace("/", " ", regex=False)
    )
    products["field_features"] = products["product_features"].map(features_to_text)
    products["field_desc"] = products["product_description"]
    products["field_all"] = (
        products["field_name"] + " " + products["field_classcat"] + " "
        + products["field_features"] + " " + products["field_desc"]
    )
    return products
