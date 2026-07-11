"""Load the trained model when available and provide a safe demo fallback."""

from pathlib import Path

import joblib
import numpy as np

from utils.preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "fake_review_model.pkl"
VECTORIZER_PATH = ROOT / "model" / "tfidf_vectorizer.pkl"


def load_prediction_assets():
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def _heuristic_prediction(text: str) -> tuple[str, float]:
    """UI fallback only; this is not a trained fake-review classifier."""
    words = text.split()
    generic = {"bagus", "mantap", "recommended", "murah", "terbaik", "jos", "oke"}
    hits = sum(word in generic for word in words)
    risk = 0.22 + (0.24 if len(words) <= 3 else 0) + min(hits * 0.12, 0.36)
    risk = float(np.clip(risk, 0.08, 0.88))
    return ("Fake", risk) if risk >= 0.5 else ("Original", 1 - risk)


def predict_review(text: object) -> tuple[str, float, str]:
    """Return prediction label, confidence, and source (model/fallback)."""
    cleaned = preprocess_text(text)
    model, vectorizer = load_prediction_assets()
    if model is None or vectorizer is None:
        label, confidence = _heuristic_prediction(cleaned)
        return label, confidence, "fallback"

    features = vectorizer.transform([cleaned])
    raw = model.predict(features)[0]
    is_fake = str(raw).lower() in {"1", "fake", "fake indicator", "true"}
    label = "Fake" if is_fake else "Original"
    confidence = float(np.max(model.predict_proba(features)[0])) if hasattr(model, "predict_proba") else 1.0
    return label, confidence, "model"
