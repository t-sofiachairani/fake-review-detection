import numpy as np
import pandas as pd

from utils.features import add_text_features, add_user_features, fit_cosine_reference, max_cosine_similarity
from utils.candidates import fit_recipe, transform_recipe
from utils.training import cross_val_scores, make_classifier
from utils.semi_supervised import select_balanced_pseudo_labels


def _labeled_frame():
    fake = ["mantap mantap murah bagus recommended"] * 10 + ["bagus bagus terbaik murah oke"] * 10
    real = ["pengiriman agak lama tapi barang sesuai deskripsi dan packing rapi"] * 10 + [
        "kualitas produk lumayan walau harga sedikit mahal menurut saya"
    ] * 10
    comments = fake + real
    labels = np.array([1] * 20 + [0] * 20)
    df = pd.DataFrame(
        {
            "comment": comments,
            "userid": [f"u{i % 7}" for i in range(len(comments))],
            "ctime": [1578476213 + i * 100 for i in range(len(comments))],
        }
    )
    frame = add_user_features(add_text_features(df))
    vec, ref = fit_cosine_reference(frame["comment_clean"])
    frame["max_cosine_similarity"] = max_cosine_similarity(frame["comment_clean"], vec, ref, exclude_self=True)
    return frame, labels


def test_cross_val_scores_keys_and_ranges():
    frame, labels = _labeled_frame()
    scores = cross_val_scores("tfidf_patterns", frame, labels, None, n_splits=5, n_repeats=2)
    for key in ["cv_f1_mean", "cv_f1_std", "cv_auc_mean", "cv_acc_mean", "cv_acc_std", "uses_bert"]:
        assert key in scores
    assert 0.0 <= scores["cv_f1_mean"] <= 1.0
    assert scores["cv_f1_std"] >= 0.0
    assert scores["uses_bert"] is False


def test_cross_val_learns_separable_signal():
    frame, labels = _labeled_frame()
    scores = cross_val_scores("tfidf_patterns", frame, labels, None, n_splits=5, n_repeats=2)
    assert scores["cv_f1_mean"] > 0.7


def test_bert_recipe_flag_and_embedding_use():
    frame, labels = _labeled_frame()
    rng = np.random.RandomState(0)
    emb = np.where(labels[:, None] == 1, 1.0, -1.0) + rng.randn(len(labels), 6) * 0.1
    scores = cross_val_scores("indobert", frame, labels, emb, n_splits=5, n_repeats=2)
    assert scores["uses_bert"] is True
    assert scores["cv_f1_mean"] > 0.7


def test_make_classifier_recipe_specific():
    assert make_classifier("tfidf_patterns").__class__.__name__ == "RandomForestClassifier"
    assert make_classifier("indobert").__class__.__name__ == "LogisticRegression"
    assert make_classifier("indobert_patterns").__class__.__name__ == "LogisticRegression"


def test_indobert_recipe_standardizes_embeddings():
    frame, _ = _labeled_frame()
    embeddings = np.arange(len(frame) * 6, dtype=float).reshape(len(frame), 6)
    artifacts = fit_recipe("indobert", frame, embeddings)
    transformed = transform_recipe(artifacts, frame, embeddings)
    assert "scaler" in artifacts
    assert np.allclose(transformed.mean(axis=0), 0.0)
    assert np.allclose(transformed.std(axis=0), 1.0)


def test_pseudo_labels_require_agreement_and_are_balanced():
    rng = np.random.RandomState(42)
    x_original = rng.normal(-2.0, 0.2, size=(20, 8))
    x_fake = rng.normal(2.0, 0.2, size=(20, 8))
    x_train = np.vstack([x_original, x_fake])
    y_train = np.array([0] * 20 + [1] * 20)
    x_pool = np.vstack([
        rng.normal(-2.0, 0.2, size=(30, 8)),
        rng.normal(2.0, 0.2, size=(30, 8)),
    ])
    selected = select_balanced_pseudo_labels(
        x_train, y_train, x_pool, confidence_threshold=0.70, max_per_class=10
    )
    assert len(selected.labels) == 20
    assert (selected.labels == 0).sum() == 10
    assert (selected.labels == 1).sum() == 10
