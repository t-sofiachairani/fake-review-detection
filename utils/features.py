"""Shared feature engineering for training AND inference.

This module is the single source of truth for how a raw review row becomes
model input. Both `train_model.py` and `utils/prediction.py` import from here,
so the features seen at training time are guaranteed identical to inference.

Pipeline (mirrors USE CASE 12.ipynb):
  1. clean_comment: lowercase + word-token join (nltk-free, deterministic)
  2. text features: review_length, positive_word_ratio, repetition_score
  3. user features: user_review_count, user_review_per_day
  4. max_cosine_similarity: leakage-aware, computed against the TRAIN corpus
  5. build_model_matrix: hstack(TF-IDF text, numeric features)
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity

# Positive-sentiment lexicon (from notebook cell 23). Generic praise words are a
# weak signal for templated / incentivised reviews.
POSITIVE_WORDS = {
    "bagus", "mantap", "baik", "recommended", "rekomendasi", "terbaik",
    "murah", "cepat", "puas", "suka", "keren", "oke", "ok", "aman",
    "rapi", "sesuai", "worth", "top",
}

# Numeric features appended to the TF-IDF matrix, in fixed order.
MODEL_FEATURE_COLS = [
    "review_length",
    "repetition_score",
    "positive_word_ratio",
    "user_review_count",
    "user_review_per_day",
    "max_cosine_similarity",
]

_TOKEN_RE = re.compile(r"\w+")


def clean_comment(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse a ctime column that may be epoch seconds or already datetime."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def add_text_features(df: pd.DataFrame, comment_col: str = "comment") -> pd.DataFrame:
    """Add comment_clean + row-local text features (no cross-row leakage)."""
    df = df.copy()
    df["comment_clean"] = df[comment_col].map(clean_comment)
    tokens = df["comment_clean"].str.split()

    df["review_length"] = tokens.map(len)
    df["positive_word_count"] = tokens.map(
        lambda words: sum(word in POSITIVE_WORDS for word in words)
    )
    df["repeated_word_count"] = tokens.map(
        lambda words: sum(count - 1 for count in Counter(words).values() if count > 1)
    )

    length = df["review_length"].replace(0, np.nan)
    df["positive_word_ratio"] = (df["positive_word_count"] / length).fillna(0.0)
    df["repetition_score"] = (df["repeated_word_count"] / length).fillna(0.0)
    return df


def add_user_features(
    df: pd.DataFrame, user_col: str = "userid", time_col: str = "ctime"
) -> pd.DataFrame:
    """Add per-user frequency features (burst / velocity signals)."""
    df = df.copy()
    if user_col not in df:
        df["user_review_count"] = 1.0
        df["user_review_per_day"] = 1.0
        return df

    when = _to_datetime(df[time_col]) if time_col in df else pd.Series(pd.NaT, index=df.index)
    review_date = when.dt.date

    df["user_review_count"] = (
        df.groupby(user_col)[user_col].transform("size").fillna(1).astype(float)
    )
    df["user_review_per_day"] = (
        df.assign(_d=review_date)
        .groupby([user_col, "_d"])[user_col]
        .transform("size")
        .fillna(1)
        .astype(float)
    )
    return df


def fit_cosine_reference(clean_texts: pd.Series):
    """Fit the cosine TF-IDF on TRAIN text and return (vectorizer, matrix)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(clean_texts)
    return vectorizer, matrix


def max_cosine_similarity(
    clean_texts: pd.Series,
    vectorizer,
    reference_matrix: csr_matrix,
    exclude_self: bool = False,
) -> np.ndarray:
    """Max cosine similarity of each text against the reference (train) corpus.

    High similarity flags near-duplicate / templated reviews. Computed against
    the TRAIN corpus only, so no test/inference text influences the reference.
    """
    if reference_matrix.shape[0] == 0:
        return np.zeros(len(clean_texts))

    vectors = vectorizer.transform(clean_texts)
    similarity = cosine_similarity(vectors, reference_matrix, dense_output=False)

    if exclude_self and similarity.shape[0] == similarity.shape[1]:
        similarity = similarity.tolil()
        similarity.setdiag(0)
        similarity = similarity.tocsr()
        similarity.eliminate_zeros()

    if similarity.shape[1] == 0:
        return np.zeros(vectors.shape[0])
    return np.asarray(similarity.max(axis=1).todense()).ravel()


def build_model_matrix(df: pd.DataFrame, model_tfidf) -> csr_matrix:
    """hstack(TF-IDF text features, numeric feature block) -> model input."""
    text_matrix = model_tfidf.transform(df["comment_clean"])
    numeric = (
        df[MODEL_FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )
    return hstack([text_matrix, csr_matrix(numeric)], format="csr")
