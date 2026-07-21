"""User-facing labels for technical model classes."""

PREDICTION_LABELS = {
    "Original": "Tampak Wajar",
    "Fake": "Perlu Ditinjau",
}


def prediction_label(value: object) -> str:
    """Return a cautious display label without changing the model value."""
    text = str(value)
    return PREDICTION_LABELS.get(text, text)
