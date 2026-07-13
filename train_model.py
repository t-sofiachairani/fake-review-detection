from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
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
    load_cache,
    save_cache,
)
from utils.features import (
    MODEL_FEATURE_COLS,
    add_text_features,
    add_user_features,
    fit_cosine_reference,
    max_cosine_similarity,
)
from utils.training import cross_val_scores
from utils.semi_supervised import (
    MAX_PSEUDO_PER_CLASS,
    TEACHER_CONFIDENCE,
    make_calibrated_lr,
    select_balanced_pseudo_labels,
)

ROOT = Path(__file__).resolve().parent
LABELED_PATH = ROOT / "train_review_only.csv"
APP_DATA_PATH = ROOT / "data" / "review_shopee.csv"
NOTEBOOK_POOL_PATH = ROOT / "notebook" / "review_shopee.csv"
MODEL_DIR = ROOT / "model"
BUNDLE_PATH = MODEL_DIR / "fake_review_model.pkl"
META_PATH = MODEL_DIR / "model_meta.json"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CACHE_VERSION = 1
PRODUCTION_RECIPE = "indobert"


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
        out["brier_score"] = float(brier_score_loss(y_true, y_proba))
        out["log_loss"] = float(log_loss(y_true, y_proba, labels=[0, 1]))
    return out


def _embed_frame(frame, embedder, cache):
    texts = frame["comment"].fillna("").astype(str).tolist()
    matrix, cache = embed_with_cache(texts, embedder, cache)
    return matrix, cache


def load_unlabeled_pool(labeled: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    source_path = NOTEBOOK_POOL_PATH if NOTEBOOK_POOL_PATH.exists() else APP_DATA_PATH
    source = pd.read_csv(source_path, low_memory=False).dropna(subset=["comment"]).copy()
    pool = _engineer(source)
    before = len(pool)
    labeled_texts = set(labeled["comment_clean"])
    overlap = pool["comment_clean"].isin(labeled_texts)
    pool = pool.loc[~overlap].copy().reset_index(drop=True)
    if pool["comment_clean"].isin(labeled_texts).any():
        raise AssertionError("labeled text remains in the unlabeled pool")
    return pool, {
        "source": str(source_path.relative_to(ROOT)),
        "rows_before_overlap_filter": int(before),
        "overlap_rows_removed": int(overlap.sum()),
        "rows_after_overlap_filter": int(len(pool)),
    }


def fit_semi_supervised_student(train, train_emb, pool_emb):
    train_labels = train["label"].to_numpy()
    teacher_artifacts = fit_recipe("indobert", train, train_emb)
    x_teacher_train = transform_recipe(teacher_artifacts, train, train_emb)
    x_teacher_pool = transform_recipe(teacher_artifacts, train.iloc[:0], pool_emb)
    pseudo = select_balanced_pseudo_labels(
        x_teacher_train,
        train_labels,
        x_teacher_pool,
        confidence_threshold=TEACHER_CONFIDENCE,
        max_per_class=MAX_PSEUDO_PER_CLASS,
    )

    combined_emb = np.vstack([train_emb, pool_emb[pseudo.positions]])
    combined_labels = np.concatenate([train_labels, pseudo.labels])
    student_artifacts = fit_recipe("indobert", train, combined_emb)
    x_student = transform_recipe(student_artifacts, train, combined_emb)
    student = make_calibrated_lr(cv=5)
    student.fit(x_student, combined_labels)
    diagnostics = {
        "teacher_confidence_threshold": TEACHER_CONFIDENCE,
        "max_pseudo_per_class": MAX_PSEUDO_PER_CLASS,
        "eligible_original": pseudo.candidate_original,
        "eligible_fake": pseudo.candidate_fake,
        "selected_original": int((pseudo.labels == 0).sum()),
        "selected_fake": int((pseudo.labels == 1).sum()),
        "selected_total": int(len(pseudo.labels)),
        "mean_joint_confidence": float(pseudo.confidence.mean()),
        "combined_training_rows": int(len(combined_labels)),
    }
    return student_artifacts, student, pseudo, diagnostics


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    labeled = load_labeled()
    labels = labeled["label"].to_numpy()
    print(f"labeled rows: {len(labeled)} | fake={int(labels.sum())}")

    embedder = IndoBertEmbedder(MODEL_NAME)
    cache = load_cache(CACHE_PATH)
    labeled_emb, cache = _embed_frame(labeled, embedder, cache)
    print(f"embedded labeled set: {labeled_emb.shape}")

    pool, pool_meta = load_unlabeled_pool(labeled)
    pool_emb, cache = _embed_frame(pool, embedder, cache)
    print(
        "unlabeled pool: "
        f"{pool_meta['rows_before_overlap_filter']} -> {len(pool)} rows "
        f"(removed overlap={pool_meta['overlap_rows_removed']})"
    )

    candidates: dict[str, dict] = {}
    for recipe in RECIPES:
        emb = labeled_emb if recipe != "tfidf_patterns" else None
        candidates[recipe] = cross_val_scores(recipe, labeled, labels, emb)
        scores = candidates[recipe]
        print(
            f"  {recipe}: f1={scores['cv_f1_mean']:.3f}±{scores['cv_f1_std']:.3f} "
            f"auc={scores['cv_auc_mean']:.3f}"
        )

    selected = PRODUCTION_RECIPE
    recipe_uses_bert = True
    print(f"SELECTED: {selected}")

    train, test = group_split(labeled)
    test_labels = test["label"].to_numpy()

    train_emb, cache = _embed_frame(train, embedder, cache)
    test_emb, cache = _embed_frame(test, embedder, cache)

    artifacts, classifier, eval_pseudo, evaluation_pseudo = fit_semi_supervised_student(
        train, train_emb, pool_emb
    )
    x_test = transform_recipe(artifacts, test, test_emb)
    prediction = classifier.predict(x_test)
    proba = classifier.predict_proba(x_test)[:, 1]
    final_metrics = report(test_labels, prediction, proba)
    print(
        "semi-supervised holdout: "
        f"pseudo={evaluation_pseudo['selected_total']} "
        f"acc={final_metrics['accuracy']:.3f} f1={final_metrics['f1']:.3f}"
    )

    full_artifacts, final_classifier, production_pseudo, production_pseudo_meta = (
        fit_semi_supervised_student(labeled, labeled_emb, pool_emb)
    )

    cosine_vectorizer, cosine_reference = fit_cosine_reference(labeled["comment_clean"])

    app_source = pd.read_csv(APP_DATA_PATH, low_memory=False)
    app_source["comment"] = app_source["comment"].fillna("").astype(str)
    app_df = _attach_cosine(_engineer(app_source))
    _, cache = _embed_frame(app_df, embedder, cache)
    save_cache(cache, CACHE_PATH)
    print(f"cached embeddings: {len(cache)} entries -> {CACHE_PATH.name}")

    confusion = confusion_matrix(test_labels, prediction, labels=[0, 1]).tolist()
    feature_description = [
        "IndoBERT embeddings (768-dim, mean-pooled)",
        "StandardScaler",
        "Calibrated Logistic Regression",
        "Balanced pseudo-labels from LR + Random Forest teacher agreement",
    ]
    honest_note = (
        "Test berlabel dikunci sebelum pseudo-labeling. Seluruh overlap teks antara data berlabel "
        "dan pool tanpa label dihapus. Pseudo-label hanya dipilih jika calibrated Logistic "
        "Regression dan Random Forest sepakat, melewati ambang keyakinan, dan jumlah kedua kelas "
        "dibuat seimbang. Evaluasi akhir hanya memakai {t} label asli yang tidak masuk training. "
        "Dengan total hanya {n} label asli, metrik tetap memiliki ketidakpastian tinggi."
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
        "embedding_text_col": "comment",
        "cosine_vectorizer": cosine_vectorizer,
        "cosine_reference": cosine_reference,
    }
    joblib.dump(bundle, BUNDLE_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected,
        "feature_recipe": "indobert_semi_supervised",
        "features": feature_description,
        "n_labeled": int(len(labeled)),
        "n_test": int(len(test)),
        "candidates": candidates,
        "final_metrics": final_metrics,
        "cv_accuracy_mean": candidates[selected]["cv_acc_mean"],
        "cv_accuracy_std": candidates[selected]["cv_acc_std"],
        "confusion_matrix": confusion,
        "confusion_labels": ["Original", "Fake"],
        "unlabeled_pool": pool_meta,
        "evaluation_pseudo_labels": evaluation_pseudo,
        "production_pseudo_labels": production_pseudo_meta,
        "training_strategy": "balanced_teacher_agreement",
        "teachers": ["calibrated_indobert_logistic_regression", "indobert_random_forest"],
        "honest_note": honest_note,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved bundle -> {BUNDLE_PATH.name} | meta -> {META_PATH.name}")


if __name__ == "__main__":
    main()
