"""User-facing labels for technical model classes."""

PREDICTION_LABELS = {
    "Original": "Original",
    "Fake": "Fake",
}


def prediction_label(value: object) -> str:
    """Return a cautious display label without changing the model value."""
    text = str(value)
    return PREDICTION_LABELS.get(text, text)
