from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128
BATCH_SIZE = 16

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "model" / "embedding_cache.pkl"


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(attention_mask, dtype=np.float64)[..., None]
    summed = (np.asarray(last_hidden_state, dtype=np.float64) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def embedding_key(model_name: str, cleaned_text: str) -> str:
    payload = f"{model_name}|{cleaned_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_cache(path: Path = CACHE_PATH) -> dict[str, np.ndarray]:
    if path.exists():
        return joblib.load(path)
    return {}


def save_cache(cache: dict[str, np.ndarray], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, path)


class IndoBertEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768))
        self._ensure_loaded()
        torch = self._torch
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self._model(**encoded)
            hidden = output.last_hidden_state.numpy()
            mask = encoded["attention_mask"].numpy()
            vectors.append(mean_pool(hidden, mask))
        return np.vstack(vectors)


def embed_with_cache(
    cleaned_texts: list[str],
    embedder: IndoBertEmbedder,
    cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cache = cache if cache is not None else load_cache()
    keys = [embedding_key(embedder.model_name, text) for text in cleaned_texts]
    missing_idx = [i for i, key in enumerate(keys) if key not in cache]
    if missing_idx:
        fresh = embedder.embed_texts([cleaned_texts[i] for i in missing_idx])
        for offset, i in enumerate(missing_idx):
            cache[keys[i]] = fresh[offset]
    matrix = np.vstack([cache[key] for key in keys]) if keys else np.zeros((0, 768))
    return matrix, cache
