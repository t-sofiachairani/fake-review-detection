import pandas as pd

from utils.features import (
    MODEL_FEATURE_COLS,
    POSITIVE_WORDS,
    add_text_features,
    add_user_features,
    clean_comment,
    fit_cosine_reference,
    max_cosine_similarity,
)


def test_clean_comment_lowercases_and_strips_punctuation():
    assert clean_comment("Bagus BANGET!!! 100% mantap.") == "bagus banget 100 mantap"


def test_clean_comment_collapses_whitespace():
    assert clean_comment("  halo    dunia \n produk  ") == "halo dunia produk"


def test_clean_comment_handles_none_and_nan():
    assert clean_comment(None) == ""
    assert clean_comment(float("nan")) == ""


def test_review_length_counts_tokens():
    df = add_text_features(pd.DataFrame({"comment": ["bagus banget mantap sekali"]}))
    assert df.loc[0, "review_length"] == 4


def test_positive_word_ratio_between_zero_and_one():
    df = add_text_features(pd.DataFrame({"comment": ["bagus mantap kursi meja"]}))
    ratio = df.loc[0, "positive_word_ratio"]
    assert 0.0 <= ratio <= 1.0
    assert ratio == 0.5


def test_repetition_score_positive_for_repeats():
    df = add_text_features(pd.DataFrame({"comment": ["bagus bagus bagus lain"]}))
    assert df.loc[0, "repetition_score"] > 0.0


def test_repetition_score_zero_for_unique_words():
    df = add_text_features(pd.DataFrame({"comment": ["satu dua tiga empat"]}))
    assert df.loc[0, "repetition_score"] == 0.0


def test_empty_comment_no_division_error():
    df = add_text_features(pd.DataFrame({"comment": [""]}))
    assert df.loc[0, "review_length"] == 0
    assert df.loc[0, "positive_word_ratio"] == 0.0
    assert df.loc[0, "repetition_score"] == 0.0


def test_add_user_features_without_userid_degrades_to_one():
    df = add_text_features(pd.DataFrame({"comment": ["bagus", "mantap"]}))
    result = add_user_features(df)
    assert (result["user_review_count"] == 1.0).all()
    assert (result["user_review_per_day"] == 1.0).all()


def test_add_user_features_counts_per_user():
    df = pd.DataFrame(
        {
            "comment": ["a", "b", "c"],
            "userid": ["u1", "u1", "u2"],
            "ctime": [1578476213, 1578476300, 1578476400],
        }
    )
    result = add_user_features(add_text_features(df))
    counts = dict(zip(result["userid"], result["user_review_count"]))
    assert counts["u1"] == 2.0
    assert counts["u2"] == 1.0


def test_positive_words_lexicon_nonempty():
    assert "bagus" in POSITIVE_WORDS
    assert len(POSITIVE_WORDS) > 5


def test_max_cosine_similarity_identical_texts_high():
    texts = pd.Series(["barang bagus sekali", "barang bagus sekali", "produk lain berbeda total"])
    vectorizer, reference = fit_cosine_reference(texts)
    scores = max_cosine_similarity(texts, vectorizer, reference)
    assert scores[0] > 0.9


def test_model_feature_cols_present_after_pipeline():
    df = pd.DataFrame(
        {"comment": ["bagus mantap", "produk lain"], "userid": ["u1", "u2"], "ctime": [1578476213, 1578476300]}
    )
    frame = add_user_features(add_text_features(df))
    for col in MODEL_FEATURE_COLS:
        if col == "max_cosine_similarity":
            continue
        assert col in frame.columns
