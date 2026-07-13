from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from utils.features import MODEL_FEATURE_COLS

RECIPES = ("tfidf_patterns", "indobert", "indobert_patterns")


def _pattern_block(frame):
    return (
        frame[MODEL_FEATURE_COLS]
        .apply(lambda col: col.astype(float))
        .fillna(0.0)
        .to_numpy()
    )


def _require_embeddings(recipe: str, embeddings) -> None:
    if embeddings is None:
        raise ValueError(f"recipe '{recipe}' requires embeddings")


def fit_recipe(recipe: str, frame, embeddings) -> dict:
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")

    if recipe == "tfidf_patterns":
        vectorizer = TfidfVectorizer()
        vectorizer.fit(frame["comment_clean"])
        return {"recipe": recipe, "vectorizer": vectorizer}

    _require_embeddings(recipe, embeddings)

    if recipe == "indobert":
        scaler = StandardScaler()
        scaler.fit(np.asarray(embeddings, dtype=np.float64))
        return {"recipe": recipe, "scaler": scaler}

    scaler = StandardScaler()
    scaler.fit(_pattern_block(frame))
    return {"recipe": recipe, "scaler": scaler}


def transform_recipe(artifacts: dict, frame, embeddings):
    recipe = artifacts["recipe"]

    if recipe == "tfidf_patterns":
        text = artifacts["vectorizer"].transform(frame["comment_clean"])
        patterns = csr_matrix(_pattern_block(frame))
        return hstack([text, patterns], format="csr")

    _require_embeddings(recipe, embeddings)
    embed = np.asarray(embeddings, dtype=np.float64)

    if recipe == "indobert":
        return artifacts["scaler"].transform(embed)

    scaled = artifacts["scaler"].transform(_pattern_block(frame))
    return np.hstack([embed, scaled])
