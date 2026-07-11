"""Visual system shared by all Streamlit pages."""

import streamlit as st


ORANGE = "#ee4d2d"
INK = "#172033"
MUTED = "#687086"


def setup_page(title: str, icon: str = "◈", show_eyebrow: bool = True) -> None:
    st.set_page_config(page_title=f"{title} · ShopAI", page_icon="🛍️", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {font-family:'DM Sans',sans-serif; color:#172033}
        .stApp {background:#f7f8fa}
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important}
        [data-testid="stHeader"] {display:none}
        .block-container {max-width:1240px;padding-top:1.2rem;padding-bottom:3rem}
        [data-testid="stMetric"] {background:#fff; border:1px solid #eceef4; border-radius:16px;
          padding:20px; box-shadow:0 6px 22px rgba(23,32,51,.05)}
        [data-testid="stMetricValue"] {color:#172033; font-weight:700}
        div.stButton > button {border-radius:10px; background:#ee4d2d; color:white; border:0}
        div.stButton > button:hover {background:#d83f20;color:white;border-color:#d83f20}
        .hero {background:linear-gradient(120deg,#ee4d2d,#ff7658); color:white; padding:38px;
          border-radius:24px; box-shadow:0 16px 40px rgba(238,77,45,.18); margin-bottom:24px}
        .hero h1 {margin:0 0 10px;font-size:2.25rem}.hero p{margin:0;opacity:.9;max-width:720px}
        .eyebrow {font-size:.75rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;color:#ee4d2d}
        .panel {background:#fff;border:1px solid #eceef4;border-radius:18px;padding:24px;margin:8px 0 18px}
        .pill {display:inline-block;padding:6px 11px;border-radius:99px;background:#fff3ef;color:#c43e24;
          font-size:.78rem;font-weight:700}
        .product-title {height:72px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;
          -webkit-box-orient:vertical;font-size:16px;font-weight:700;line-height:1.45;
          color:#172033;margin:8px 0 4px}
        .product-meta {height:26px;color:#8a91a3;font-size:13px;display:flex;align-items:center}
        .trust-score {height:42px;box-sizing:border-box;background:#fff3ef;color:#c43e24;
          padding:10px;border:1px solid #ffc3b5;border-radius:8px;font-size:12px;font-weight:700;
          display:flex;align-items:center;justify-content:space-between}
        h1,h2,h3 {letter-spacing:-.025em}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if show_eyebrow:
        st.markdown(f'<div class="eyebrow">{icon} &nbsp; {title}</div>', unsafe_allow_html=True)


def marketplace_header() -> None:
    """Reserved hook for a future marketplace announcement bar."""
    return None


def prediction_notice(df) -> None:
    if "fallback" in set(df.get("prediction_source", [])):
        st.info("Mode demo aktif: indikator risiko menggunakan aturan sederhana karena model .pkl belum tersedia.")
