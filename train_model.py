from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils.candidates import RECIPES, fit_recipe, transform_recipe
from utils.embeddings import (
    CACHE_PATH,
    MODEL_NAME,
    IndoBertEmbedder,
    embed_with_cache,
    save_cache,
)
from utils.features import (
    MODEL_FEATURE_COLS,
    add_text_features,
    add_user_features,
    fit_cosine_reference,
    max_cosine_similarity,
)
from utils.model_selection import select_model
from utils.training import cross_val_scores, make_classifier

ROOT = Path(__file__).resolve().parent
LABELED_PATH = ROOT / "train_review_only.csv"
APP_DATA_PATH = ROOT / "data" / "review_shopee.csv"
MODEL_DIR = ROOT / "model"
BUNDLE_PATH = MODEL_DIR / "fake_review_model.pkl"
META_PATH = MODEL_DIR / "model_meta.json"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CACHE_VERSION = 1


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    return add_user_features(add_text_features(df))


def _attach_cosine(frame: pd.DataFrame) -> pd.DataFrame:
    vectorizer, reference = fit_cosine_reference(frame["comment_clean"])
    frame = frame.copy()
    frame["max_cosine_similarity"] = max_cosine_similarity(
        frame["comment_clean"], vectorizer, reference, exclude_self=True
    )
    return frame


def load_labeled() -> pd.DataFrame:
    df = pd.read_csv(LABELED_PATH, low_memory=False).dropna(subset=["comment"])
    df["label"] = df["fakeornot"].astype(str).str.lower().map({"original": 0, "fake": 1})
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return _attach_cosine(_engineer(df))


def group_split(df: pd.DataFrame):
    groups = df.groupby("comment_clean", as_index=False).agg(
        label=("label", lambda x: x.mode().iat[0])
    )
    train_texts, test_texts = train_test_split(
        groups["comment_clean"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=groups["label"],
    )
    train = df[df["comment_clean"].isin(set(train_texts))].copy()
    test = df[df["comment_clean"].isin(set(test_texts))].copy()
    return train, test


def report(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support": int(len(y_true)),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return out


def _embed_frame(frame, embedder, cache):
    matrix, cache = embed_with_cache(frame["comment_clean"].tolist(), embedder, cache)
    return matrix, cache


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    labeled = load_labeled()
    labels = labeled["label"].to_numpy()
    print(f"labeled rows: {len(labeled)} | fake={int(labels.sum())}")

    embedder = IndoBertEmbedder(MODEL_NAME)
    cache = {}
    labeled_emb, cache = _embed_frame(labeled, embedder, cache)
    print(f"embedded labeled set: {labeled_emb.shape}")

    candidates: dict[str, dict] = {}
    for recipe in RECIPES:
        emb = labeled_emb if recipe != "tfidf_patterns" else None
        candidates[recipe] = cross_val_scores(recipe, labeled, labels, emb)
        scores = candidates[recipe]
        print(
            f"  {recipe}: f1={scores['cv_f1_mean']:.3f}±{scores['cv_f1_std']:.3f} "
            f"auc={scores['cv_auc_mean']:.3f}"
        )

    selected = select_model(candidates)
    recipe_uses_bert = selected != "tfidf_patterns"
    print(f"SELECTED: {selected}")

    train, test = group_split(labeled)
    train_labels = train["label"].to_numpy()
    test_labels = test["label"].to_numpy()

    if recipe_uses_bert:
        train_emb, cache = _embed_frame(train, embedder, cache)
        test_emb, cache = _embed_frame(test, embedder, cache)
    else:
        train_emb = test_emb = None

    artifacts = fit_recipe(selected, train, train_emb)
    x_train = transform_recipe(artifacts, train, train_emb)
    x_test = transform_recipe(artifacts, test, test_emb)
    classifier = make_classifier(selected)
    classifier.fit(x_train, train_labels)
    proba = classifier.predict_proba(x_test)[:, 1] if hasattr(classifier, "predict_proba") else None
    final_metrics = report(test_labels, classifier.predict(x_test), proba)
    print(f"holdout: acc={final_metrics['accuracy']:.3f} f1={final_metrics['f1']:.3f}")

    full_artifacts = fit_recipe(selected, labeled, labeled_emb if recipe_uses_bert else None)
    x_full = transform_recipe(full_artifacts, labeled, labeled_emb if recipe_uses_bert else None)
    final_classifier = make_classifier(selected)
    final_classifier.fit(x_full, labels)

    cosine_vectorizer, cosine_reference = fit_cosine_reference(labeled["comment_clean"])

    if recipe_uses_bert:
        app_df = _attach_cosine(_engineer(pd.read_csv(APP_DATA_PATH, low_memory=False).dropna(subset=["comment"])))
        _, cache = _embed_frame(app_df, embedder, cache)
        save_cache(cache, CACHE_PATH)
        print(f"cached embeddings: {len(cache)} entries -> {CACHE_PATH.name}")

    confusion = confusion_matrix(test_labels, classifier.predict(x_test)).tolist()
    feature_description = {
        "tfidf_patterns": ["TF-IDF", *MODEL_FEATURE_COLS],
        "indobert": ["IndoBERT embeddings (768-dim, mean-pooled)"],
        "indobert_patterns": ["IndoBERT embeddings (768-dim)", *MODEL_FEATURE_COLS],
    }[selected]
    honest_note = (
        "Model dipilih via RepeatedStratifiedKFold (5x3 = 75 fold) berdasarkan F1 rata-rata, "
        "dengan aturan parsimoni: bila beberapa kandidat setara dalam 1 standar deviasi, "
        "dipilih yang paling sederhana dan bebas kebocoran fitur. Recipe '{r}' terpilih. "
        "Dengan hanya {n} data berlabel (uji {t} review), metrik holdout bervariasi; "
        "gunakan skor CV sebagai estimasi utama. Data latih didominasi review akun digital/Netflix, "
        "sehingga review domain lain bisa kurang akurat."
    ).format(r=selected, n=len(labeled), t=len(test))

    bundle = {
        "recipe": selected,
        "artifacts": full_artifacts,
        "classifier": final_classifier,
        "feature_cols": MODEL_FEATURE_COLS,
        "model_name": MODEL_NAME,
        "cache_version": CACHE_VERSION,
        "threshold": 0.5,
        "label_map": {0: "Original", 1: "Fake"},
        "uses_bert": recipe_uses_bert,
        "cosine_vectorizer": cosine_vectorizer,
        "cosine_reference": cosine_reference,
    }
    joblib.dump(bundle, BUNDLE_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected,
        "feature_recipe": selected,
        "features": feature_description,
        "n_labeled": int(len(labeled)),
        "n_test": int(len(test)),
        "candidates": candidates,
        "final_metrics": final_metrics,
        "cv_accuracy_mean": candidates[selected]["cv_acc_mean"],
        "cv_accuracy_std": candidates[selected]["cv_acc_std"],
        "confusion_matrix": confusion,
        "confusion_labels": ["Original", "Fake"],
        "honest_note": honest_note,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved bundle -> {BUNDLE_PATH.name} | meta -> {META_PATH.name}")


if __name__ == "__main__":
    main()
