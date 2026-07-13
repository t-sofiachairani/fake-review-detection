import numpy as np
import pandas as pd
import pytest

from utils.candidates import RECIPES, fit_recipe, transform_recipe
from utils.features import (
    MODEL_FEATURE_COLS,
    add_text_features,
    add_user_features,
    fit_cosine_reference,
    max_cosine_similarity,
)


def _frame():
    df = pd.DataFrame(
        {
            "comment": [
                "barang bagus mantap sekali recommended",
                "pengiriman lama tapi kualitas sesuai",
                "mantap mantap mantap murah",
                "produk biasa saja tidak istimewa",
                "keren banget suka sekali puas",
                "kurang memuaskan agak kecewa",
            ],
            "userid": ["u1", "u2", "u1", "u3", "u2", "u3"],
            "ctime": [1578476213, 1578476300, 1578476400, 1578476500, 1578476600, 1578476700],
        }
    )
    frame = add_user_features(add_text_features(df))
    vectorizer, reference = fit_cosine_reference(frame["comment_clean"])
    frame["max_cosine_similarity"] = max_cosine_similarity(
        frame["comment_clean"], vectorizer, reference, exclude_self=True
    )
    return frame


def test_recipes_registry():
    assert set(RECIPES) == {"tfidf_patterns", "indobert", "indobert_patterns"}


def test_tfidf_patterns_matrix_has_no_nan_and_rows_match():
    frame = _frame()
    artifacts = fit_recipe("tfidf_patterns", frame, None)
    matrix = transform_recipe(artifacts, frame, None)
    assert matrix.shape[0] == len(frame)
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    assert not np.isnan(dense).any()


def test_indobert_patterns_width_is_embed_plus_patterns():
    frame = _frame()
    emb = np.random.RandomState(0).randn(len(frame), 8)
    artifacts = fit_recipe("indobert_patterns", frame, emb)
    matrix = transform_recipe(artifacts, frame, emb)
    assert matrix.shape == (len(frame), 8 + len(MODEL_FEATURE_COLS))


def test_indobert_only_width_is_embed_dim():
    frame = _frame()
    emb = np.random.RandomState(1).randn(len(frame), 8)
    artifacts = fit_recipe("indobert", frame, emb)
    matrix = transform_recipe(artifacts, frame, emb)
    assert matrix.shape == (len(frame), 8)


def test_bert_recipe_requires_embeddings():
    frame = _frame()
    with pytest.raises(ValueError):
        fit_recipe("indobert", frame, None)


def test_scaler_fit_on_train_applied_to_new_rows():
    frame = _frame()
    artifacts = fit_recipe("indobert_patterns", frame, np.zeros((len(frame), 4)))
    single = frame.iloc[[0]].copy()
    matrix = transform_recipe(artifacts, single, np.zeros((1, 4)))
    assert matrix.shape == (1, 4 + len(MODEL_FEATURE_COLS))
