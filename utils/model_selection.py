from __future__ import annotations

BERT_VARIANCE_LIMIT = 0.10


def _is_eligible(scores: dict) -> bool:
    if scores.get("uses_bert") and scores.get("cv_f1_std", 0.0) > BERT_VARIANCE_LIMIT:
        return False
    return True


def _parsimony_key(item: tuple[str, dict]) -> tuple[int, float, float]:
    scores = item[1]
    return (
        scores.get("n_pattern_features", 0),
        -scores.get("cv_f1_mean", 0.0),
        -scores.get("cv_auc_mean", 0.0),
    )


def select_model(candidates: dict[str, dict]) -> str:
    if not candidates:
        raise ValueError("no candidates to select from")

    eligible = [item for item in candidates.items() if _is_eligible(item[1])]
    pool = eligible if eligible else list(candidates.items())

    best_name, best_scores = max(
        pool, key=lambda item: (item[1].get("cv_f1_mean", 0.0), item[1].get("cv_auc_mean", 0.0))
    )
    band = best_scores.get("cv_f1_mean", 0.0) - best_scores.get("cv_f1_std", 0.0)
    within_band = [item for item in pool if item[1].get("cv_f1_mean", 0.0) >= band]

    return min(within_band, key=_parsimony_key)[0]
