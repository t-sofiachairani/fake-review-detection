from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.candidates import transform_recipe
from utils.features import (
    add_text_features,
    add_user_features,
    clean_comment,
    max_cosine_similarity,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "fake_review_model.pkl"
META_PATH = ROOT / "model" / "model_meta.json"


def load_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def load_meta() -> dict | None:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return None


def _heuristic_prediction(text: str) -> tuple[str, float]:
    words = text.split()
    generic = {"bagus", "mantap", "recommended", "murah", "terbaik", "jos", "oke"}
    hits = sum(word in generic for word in words)
    risk = 0.22 + (0.24 if len(words) <= 3 else 0) + min(hits * 0.12, 0.36)
    risk = float(np.clip(risk, 0.08, 0.88))
    return ("Fake", risk) if risk >= 0.5 else ("Original", 1 - risk)


def _attach_features(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    frame = add_user_features(add_text_features(df))
    frame["max_cosine_similarity"] = max_cosine_similarity(
        frame["comment_clean"], bundle["cosine_vectorizer"], bundle["cosine_reference"]
    )
    return frame


def _embeddings_for(bundle: dict, cleaned_texts: list[str]) -> np.ndarray | None:
    if not bundle.get("uses_bert"):
        return None
    from utils.embeddings import embedding_key, load_cache

    cache = load_cache()
    keys = [embedding_key(bundle["model_name"], text) for text in cleaned_texts]
    missing = [key not in cache for key in keys]
    if any(missing):
        from utils.embeddings import IndoBertEmbedder, embed_with_cache, save_cache

        embedder = IndoBertEmbedder(bundle["model_name"])
        _, cache = embed_with_cache(cleaned_texts, embedder, cache)
        save_cache(cache)
    return np.vstack([cache[key] for key in keys])


def _score(bundle: dict, frame: pd.DataFrame, embeddings) -> tuple[np.ndarray, np.ndarray]:
    matrix = transform_recipe(bundle["artifacts"], frame, embeddings)
    classifier = bundle["classifier"]
    if hasattr(classifier, "predict_proba"):
        fake_proba = classifier.predict_proba(matrix)[:, 1]
    else:
        fake_proba = classifier.predict(matrix).astype(float)
    labels = (fake_proba >= bundle.get("threshold", 0.5)).astype(int)
    return labels, fake_proba


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    bundle = load_bundle()

    if bundle is None:
        scored = result["comment"].map(lambda text: _heuristic_prediction(clean_comment(text)))
        result[["prediction", "confidence"]] = pd.DataFrame(scored.tolist(), index=result.index)
        result["fake_probability"] = np.where(
            result["prediction"].eq("Fake"), result["confidence"], 1 - result["confidence"]
        )
        result["prediction_source"] = "fallback"
        return result

    frame = _attach_features(bundle, result)
    embeddings = _embeddings_for(bundle, frame["comment_clean"].tolist())
    labels, fake_proba = _score(bundle, frame, embeddings)
    label_map = bundle["label_map"]
    result["prediction"] = [label_map[int(label)] for label in labels]
    result["fake_probability"] = fake_proba
    result["confidence"] = np.where(labels == 1, fake_proba, 1 - fake_proba)
    result["prediction_source"] = "model"
    return result


def predict_review(text: object) -> tuple[str, float, str]:
    bundle = load_bundle()
    if bundle is None:
        label, confidence = _heuristic_prediction(clean_comment(text))
        return label, confidence, "fallback"

    frame = _attach_features(bundle, pd.DataFrame({"comment": [text]}))
    embeddings = _embeddings_for(bundle, frame["comment_clean"].tolist())
    labels, fake_proba = _score(bundle, frame, embeddings)
    label_int = int(labels[0])
    confidence = float(fake_proba[0] if label_int == 1 else 1 - fake_proba[0])
    return bundle["label_map"][label_int], confidence, "model"


def explain_review(text: object) -> dict:
    bundle = load_bundle()
    if bundle is None:
        frame = add_user_features(add_text_features(pd.DataFrame({"comment": [text]})))
        frame["max_cosine_similarity"] = 0.0
        label, confidence = _heuristic_prediction(clean_comment(text))
        fake_probability = confidence if label == "Fake" else 1 - confidence
    else:
        frame = _attach_features(bundle, pd.DataFrame({"comment": [text]}))
        embeddings = _embeddings_for(bundle, frame["comment_clean"].tolist())
        _, fake_proba = _score(bundle, frame, embeddings)
        fake_probability = float(fake_proba[0])

    row = frame.iloc[0]
    signals = {
        "review_length": int(row["review_length"]),
        "positive_word_ratio": float(row["positive_word_ratio"]),
        "repetition_score": float(row["repetition_score"]),
        "user_review_count": float(row["user_review_count"]),
        "user_review_per_day": float(row["user_review_per_day"]),
        "max_cosine_similarity": float(row["max_cosine_similarity"]),
    }
    top_signals = sorted(signals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:4]
    return {"fake_probability": fake_probability, **signals, "top_signals": top_signals}
