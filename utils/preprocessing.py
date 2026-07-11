"""Text preprocessing used by the prediction pipeline."""

import re


def preprocess_text(text: object) -> str:
    """Lowercase text, remove punctuation, and normalize whitespace."""
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return " ".join(value.split())
