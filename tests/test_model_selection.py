import pytest

from utils.model_selection import select_model


def test_picks_highest_mean_f1_when_stable():
    candidates = {
        "tfidf_patterns": {"cv_f1_mean": 0.89, "cv_f1_std": 0.05, "cv_auc_mean": 0.90, "uses_bert": False},
        "indobert": {"cv_f1_mean": 0.92, "cv_f1_std": 0.06, "cv_auc_mean": 0.93, "uses_bert": True},
    }
    assert select_model(candidates) == "indobert"


def test_bert_disqualified_when_variance_too_high():
    candidates = {
        "tfidf_patterns": {"cv_f1_mean": 0.89, "cv_f1_std": 0.05, "cv_auc_mean": 0.90, "uses_bert": False},
        "indobert": {"cv_f1_mean": 0.95, "cv_f1_std": 0.14, "cv_auc_mean": 0.96, "uses_bert": True},
    }
    assert select_model(candidates) == "tfidf_patterns"


def test_non_bert_high_variance_still_eligible():
    candidates = {
        "tfidf_patterns": {"cv_f1_mean": 0.91, "cv_f1_std": 0.15, "cv_auc_mean": 0.90, "uses_bert": False},
        "indobert": {"cv_f1_mean": 0.88, "cv_f1_std": 0.04, "cv_auc_mean": 0.89, "uses_bert": True},
    }
    assert select_model(candidates) == "tfidf_patterns"


def test_roc_auc_breaks_f1_tie():
    candidates = {
        "a": {"cv_f1_mean": 0.90, "cv_f1_std": 0.05, "cv_auc_mean": 0.91, "uses_bert": False},
        "b": {"cv_f1_mean": 0.90, "cv_f1_std": 0.05, "cv_auc_mean": 0.94, "uses_bert": False},
    }
    assert select_model(candidates) == "b"


def test_all_bert_disqualified_falls_back_to_best_eligible():
    candidates = {
        "tfidf_patterns": {"cv_f1_mean": 0.80, "cv_f1_std": 0.05, "cv_auc_mean": 0.82, "uses_bert": False},
        "indobert": {"cv_f1_mean": 0.99, "cv_f1_std": 0.20, "cv_auc_mean": 0.99, "uses_bert": True},
        "indobert_patterns": {"cv_f1_mean": 0.98, "cv_f1_std": 0.11, "cv_auc_mean": 0.99, "uses_bert": True},
    }
    assert select_model(candidates) == "tfidf_patterns"


def test_empty_raises():
    with pytest.raises(ValueError):
        select_model({})


def test_parsimony_prefers_fewer_features_within_one_se():
    candidates = {
        "indobert_patterns": {"cv_f1_mean": 0.914, "cv_f1_std": 0.049, "cv_auc_mean": 0.973, "uses_bert": True, "n_pattern_features": 6},
        "indobert": {"cv_f1_mean": 0.911, "cv_f1_std": 0.049, "cv_auc_mean": 0.974, "uses_bert": True, "n_pattern_features": 0},
        "tfidf_patterns": {"cv_f1_mean": 0.901, "cv_f1_std": 0.078, "cv_auc_mean": 0.964, "uses_bert": False, "n_pattern_features": 6},
    }
    assert select_model(candidates) == "indobert"


def test_parsimony_does_not_override_meaningful_gap():
    candidates = {
        "indobert_patterns": {"cv_f1_mean": 0.95, "cv_f1_std": 0.02, "cv_auc_mean": 0.97, "uses_bert": True, "n_pattern_features": 6},
        "indobert": {"cv_f1_mean": 0.80, "cv_f1_std": 0.02, "cv_auc_mean": 0.82, "uses_bert": True, "n_pattern_features": 0},
    }
    assert select_model(candidates) == "indobert_patterns"
