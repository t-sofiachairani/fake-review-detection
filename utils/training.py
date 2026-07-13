from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from utils.candidates import fit_recipe, transform_recipe

RANDOM_STATE = 42
BERT_RECIPES = {"indobert", "indobert_patterns"}


RECIPE_PATTERN_COUNTS = {"tfidf_patterns": 6, "indobert": 0, "indobert_patterns": 6}


def make_classifier(recipe: str):
    if recipe == "tfidf_patterns":
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)


def _slice_embeddings(embeddings, index) -> np.ndarray | None:
    if embeddings is None:
        return None
    return np.asarray(embeddings)[index]


def cross_val_scores(
    recipe: str,
    frame,
    labels,
    embeddings,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> dict:
    labels = np.asarray(labels)
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    f1s: list[float] = []
    aucs: list[float] = []
    accs: list[float] = []

    for train_idx, test_idx in splitter.split(frame, labels):
        train_frame = frame.iloc[train_idx]
        test_frame = frame.iloc[test_idx]
        train_emb = _slice_embeddings(embeddings, train_idx)
        test_emb = _slice_embeddings(embeddings, test_idx)

        artifacts = fit_recipe(recipe, train_frame, train_emb)
        x_train = transform_recipe(artifacts, train_frame, train_emb)
        x_test = transform_recipe(artifacts, test_frame, test_emb)

        classifier = make_classifier(recipe)
        classifier.fit(x_train, labels[train_idx])
        predicted = classifier.predict(x_test)
        f1s.append(f1_score(labels[test_idx], predicted, zero_division=0))
        accs.append(accuracy_score(labels[test_idx], predicted))
        if len(set(labels[test_idx])) > 1 and hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(x_test)[:, 1]
            aucs.append(roc_auc_score(labels[test_idx], proba))

    return {
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
        "cv_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        "cv_acc_mean": float(np.mean(accs)),
        "cv_acc_std": float(np.std(accs)),
        "uses_bert": recipe in BERT_RECIPES,
        "n_pattern_features": RECIPE_PATTERN_COUNTS.get(recipe, 0),
    }
