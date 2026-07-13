import sys

import pandas as pd

from utils.data import load_data
from utils.prediction import explain_review, load_bundle, predict_review


def test_bundle_loads_with_recipe():
    bundle = load_bundle()
    assert bundle is not None
    assert bundle["recipe"] in {"tfidf_patterns", "indobert", "indobert_patterns"}


def test_batch_predictions_have_contract_columns():
    df = load_data()
    for col in ["prediction", "confidence", "fake_probability", "prediction_source"]:
        assert col in df.columns
    assert set(df["prediction"].unique()) <= {"Fake", "Original"}
    assert df["confidence"].between(0.0, 1.0).all()
    assert (df["prediction_source"] == "model").all()


def test_batch_path_is_torch_free():
    sys.modules.pop("torch", None)
    load_data()
    assert "torch" not in sys.modules


def test_predict_review_returns_triplet():
    label, confidence, source = predict_review("barang bagus mantap sekali recommended murah")
    assert label in {"Fake", "Original"}
    assert 0.0 <= confidence <= 1.0
    assert source == "model"


def test_explain_review_signal_breakdown():
    result = explain_review("mantap mantap mantap bagus murah recommended")
    assert 0.0 <= result["fake_probability"] <= 1.0
    for key in ["review_length", "positive_word_ratio", "repetition_score", "max_cosine_similarity"]:
        assert key in result
    assert isinstance(result["top_signals"], list)
