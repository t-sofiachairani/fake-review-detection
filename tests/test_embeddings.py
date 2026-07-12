import numpy as np

from utils.embeddings import embedding_key, mean_pool


def test_mean_pool_ignores_padding():
    last_hidden = np.array(
        [
            [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],
        ]
    )
    attention_mask = np.array([[1, 1, 0]])
    pooled = mean_pool(last_hidden, attention_mask)
    assert pooled.shape == (1, 2)
    np.testing.assert_allclose(pooled[0], [2.0, 2.0])


def test_mean_pool_all_tokens():
    last_hidden = np.array([[[2.0, 4.0], [4.0, 8.0]]])
    attention_mask = np.array([[1, 1]])
    pooled = mean_pool(last_hidden, attention_mask)
    np.testing.assert_allclose(pooled[0], [3.0, 6.0])


def test_mean_pool_empty_mask_no_nan():
    last_hidden = np.array([[[1.0, 1.0], [2.0, 2.0]]])
    attention_mask = np.array([[0, 0]])
    pooled = mean_pool(last_hidden, attention_mask)
    assert not np.isnan(pooled).any()


def test_embedding_key_includes_model_name():
    key_a = embedding_key("model-a", "teks bersih")
    key_b = embedding_key("model-b", "teks bersih")
    assert key_a != key_b


def test_embedding_key_stable_for_same_input():
    assert embedding_key("m", "halo dunia") == embedding_key("m", "halo dunia")


def test_embedding_key_is_hex_sha256():
    key = embedding_key("m", "x")
    assert len(key) == 64
    int(key, 16)
