"""Marketplace-style entry page for ShopAI."""

from html import escape

import streamlit as st

from utils.data import load_data
from utils.ui import (
    icon,
    marketplace_header,
    page_bounds,
    page_controls,
    product_image,
    setup_page,
)


setup_page("Marketplace", "AI-assisted shopping")
df = load_data()
marketplace_header()

query = st.text_input(
    "Cari produk",
    placeholder="Cari produk yang ingin dibeli...",
    key="live_product_search",
    label_visibility="collapsed",
)
st.divider()

catalog = (
    df.groupby(["item_id", "shop_id", "product_title"], as_index=False)
    .agg(
        rating=("rating_star", "mean"),
        total_review=("comment", "size"),
        risk_ratio=("prediction", lambda values: values.eq("Fake").mean()),
    )
    .sort_values(["total_review", "rating"], ascending=False)
)
catalog["ai_trust_score"] = ((1 - catalog["risk_ratio"]) * 100).round().astype(int)
if query.strip():
    catalog = catalog[catalog["product_title"].str.contains(query.strip(), case=False, na=False)]

filter_col, result_col = st.columns([1.15, 4], gap="large")
with filter_col:
    with st.container(border=True):
        st.markdown(f'<div style="font-size:1.05rem;font-weight:700;margin:0 0 6px">{icon("verified_user")} ShopAI Trust</div>', unsafe_allow_html=True)
        st.caption("Saring berdasarkan AI Trust Score produk")
        min_trust = st.slider(
            "Minimum AI Trust Score",
            0,
            100,
            0,
            format="%d/100",
            help="Hanya tampilkan produk dengan AI Trust Score sama dengan atau di atas batas ini.",
        )
    st.caption("Indikator AI bukan keputusan absolut mengenai keaslian review.")

catalog = catalog[catalog["ai_trust_score"].ge(min_trust)]
with result_col:
    top_left, top_right = st.columns([2, 1])
    top_left.markdown(f"### {'Hasil pencarian' if query else 'Produk populer'}")
    top_right.caption(f"{len(catalog):,} produk ditemukan")

    if catalog.empty:
        st.warning("Produk tidak ditemukan. Coba gunakan kata kunci yang lebih singkat.")
    else:
        start, end = page_bounds(len(catalog), key="catalog", page_size=12)
        rows = catalog.iloc[start:end].to_dict("records")
        for row_start in range(0, len(rows), 3):
            columns = st.columns(3)
            for column, item in zip(columns, rows[row_start : row_start + 3]):
                with column.container(border=True):
                    st.markdown(
                        product_image(item["item_id"], item["shop_id"]),
                        unsafe_allow_html=True,
                    )
                    title = escape(str(item["product_title"]))
                    st.markdown(
                        f'<div class="product-title" title="{title}">{title}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="product-meta">{icon("star")} {item["rating"]:.1f} · {item["total_review"]:,} review</div>',
                        unsafe_allow_html=True,
                    )
                    score = int(item["ai_trust_score"])
                    st.markdown(
                        f'<div class="trust-row"><span>{icon("shield")} AI TRUST SCORE</span><b>{score}</b></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Lihat produk", key=f"open_{item['item_id']}_{item['shop_id']}", use_container_width=True):
                        st.session_state["selected_item_id"] = str(item["item_id"])
                        st.session_state["selected_shop_id"] = str(item["shop_id"])
                        st.switch_page("pages/1_Product_Dashboard.py")
        page_controls(len(catalog), key="catalog", page_size=12)
