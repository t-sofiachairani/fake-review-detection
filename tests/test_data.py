import pandas as pd

from utils.data import clean_data


def test_clean_data_removes_empty_and_whitespace_reviews():
    frame = pd.DataFrame(
        {
            "item_id": ["1", "1", "1", "1"],
            "shop_id": ["2", "2", "2", "2"],
            "userid": ["a", "b", "c", "d"],
            "comment": ["review valid", "", "   ", None],
            "product_title": ["Produk"] * 4,
            "rating_star": [5, 5, 5, 5],
            "ctime": [1_600_000_000] * 4,
        }
    )

    cleaned = clean_data(frame)

    assert cleaned["comment"].tolist() == ["review valid"]
