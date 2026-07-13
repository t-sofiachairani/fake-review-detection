from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42
TEACHER_CONFIDENCE = 0.80
MAX_PSEUDO_PER_CLASS = 100


@dataclass
class PseudoSelection:
    positions: np.ndarray
    labels: np.ndarray
    lr_probability: np.ndarray
    rf_probability: np.ndarray
    confidence: np.ndarray
    candidate_original: int
    candidate_fake: int

    @property
    def per_class(self) -> int:
        return int(len(self.labels) // 2)


def make_calibrated_lr(cv: int = 3) -> CalibratedClassifierCV:
    base = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=cv)


def select_balanced_pseudo_labels(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pool: np.ndarray,
    confidence_threshold: float = TEACHER_CONFIDENCE,
    max_per_class: int = MAX_PSEUDO_PER_CLASS,
) -> PseudoSelection:
    """Select balanced pseudo-labels only where calibrated LR and RF agree."""
    lr_teacher = make_calibrated_lr(cv=3)
    rf_teacher = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lr_teacher.fit(x_train, y_train)
    rf_teacher.fit(x_train, y_train)

    lr_probability = lr_teacher.predict_proba(x_pool)[:, 1]
    rf_probability = rf_teacher.predict_proba(x_pool)[:, 1]
    lr_label = (lr_probability >= 0.5).astype(int)
    rf_label = (rf_probability >= 0.5).astype(int)
    agreed = lr_label == rf_label

    agreed_label = lr_label
    lr_confidence = np.where(agreed_label == 1, lr_probability, 1 - lr_probability)
    rf_confidence = np.where(agreed_label == 1, rf_probability, 1 - rf_probability)
    joint_confidence = np.minimum(lr_confidence, rf_confidence)
    eligible = agreed & (lr_confidence >= confidence_threshold) & (
        rf_confidence >= confidence_threshold
    )

    by_class = {
        label: np.flatnonzero(eligible & (agreed_label == label)) for label in (0, 1)
    }
    per_class = min(len(by_class[0]), len(by_class[1]), max_per_class)
    if per_class == 0:
        raise ValueError(
            "teacher agreement did not produce high-confidence candidates for both classes"
        )

    selected: list[np.ndarray] = []
    for label in (0, 1):
        candidates = by_class[label]
        order = np.argsort(joint_confidence[candidates])[::-1]
        selected.append(candidates[order[:per_class]])
    positions = np.concatenate(selected)
    labels = agreed_label[positions]

    return PseudoSelection(
        positions=positions,
        labels=labels,
        lr_probability=lr_probability[positions],
        rf_probability=rf_probability[positions],
        confidence=joint_confidence[positions],
        candidate_original=int(len(by_class[0])),
        candidate_fake=int(len(by_class[1])),
    )
