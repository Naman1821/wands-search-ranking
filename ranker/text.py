"""Text normalisation shared by every component (index, queries, classifier).

The same function is applied to product text and to queries, so it matters
more that it is consistent than that it is linguistically perfect.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+")
_NOISE_VALUES = {"yes", "no", "n/a", "na", "none", "other", "not applicable"}


def strip_accents(text: str) -> str:
    """'Décor' -> 'Decor', so accented and plain spellings match."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_token(tok: str) -> str:
    """Light plural folding so 'chairs' matches 'chair' and 'benches' matches 'bench'."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith(("ches", "shes", "sses", "xes", "zes")):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text) -> list[str]:
    """Lower-case, strip accents, split on non-alphanumerics, fold plurals."""
    if not isinstance(text, str) or not text:
        return []
    return [normalize_token(t) for t in _TOKEN_RE.findall(strip_accents(text.lower()))]


def features_to_text(raw) -> str:
    """Keep only the textual values of a 'key:value|key:value' attribute string.

    Numeric values (dimensions, weights) and yes/no flags are dropped because
    they add noise to lexical matching; values such as 'rubberwood',
    'modern' or 'brushed nickel' are kept.
    """
    if not isinstance(raw, str):
        return ""
    values = []
    for part in raw.split("|"):
        if ":" not in part:
            continue
        value = part.split(":", 1)[1].strip()
        if not value or value.lower() in _NOISE_VALUES or _NUMERIC_RE.fullmatch(value):
            continue
        values.append(value)
    return " ".join(values)
