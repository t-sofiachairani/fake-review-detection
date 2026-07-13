"""Reusable loading, cleaning, filtering, and metric functions."""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.prediction import MODEL_PATH, predict_dataframe


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "review_shopee.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().dropna(how="all")
    for col in ["item_id", "shop_id", "userid"]:
        if col in df:
            df[col] = df[col].astype("string").str.replace(r"\.0$", "", regex=True)
    df["comment"] = df.get("comment", "").fillna("").astype(str)
    df["product_title"] = df.get("product_title", "Produk tanpa nama").fillna("Produk tanpa nama")
    df["rating_star"] = pd.to_numeric(df.get("rating_star", 0), errors="coerce")
    df["ctime"] = pd.to_datetime(df.get("ctime"), unit="s", errors="coerce")
    has_review = df["comment"].str.strip().ne("")
    valid = df["item_id"].notna() & df["shop_id"].notna() & has_review
    return df.loc[valid].reset_index(drop=True)


@st.cache_data(show_spinner="Menyiapkan data review...")
def load_data(path: str = str(DATA_PATH)) -> pd.DataFrame:
    df = clean_data(pd.read_csv(path, low_memory=False))
    if MODEL_PATH.exists():
        return predict_dataframe(df)

    label_col = next((c for c in ["fakeornot", "prediction"] if c in df.columns), None)
    if label_col:
        fake_values = {"1", "fake", "fake indicator", "true"}
        df["prediction"] = df[label_col].astype(str).str.lower().map(
            lambda value: "Fake" if value in fake_values else "Original"
        )
        df["confidence"] = 1.0
        df["prediction_source"] = "dataset"
        return df

    return predict_dataframe(df)


def calculate_metrics(df: pd.DataFrame) -> dict[str, float]:
    total = len(df)
    fake = int(df["prediction"].eq("Fake").sum()) if total else 0
    return {
        "reviews": total,
        "products": df["item_id"].nunique(),
        "sellers": df["shop_id"].nunique(),
        "users": df["userid"].nunique(),
        "fake_pct": fake / total * 100 if total else 0,
        "original_pct": (total - fake) / total * 100 if total else 0,
    }


def filter_product(df: pd.DataFrame, product_title: str) -> pd.DataFrame:
    return df[df["product_title"].eq(product_title)].copy()


def filter_seller(df: pd.DataFrame, shop_id: str) -> pd.DataFrame:
    return df[df["shop_id"].eq(str(shop_id))].copy()
