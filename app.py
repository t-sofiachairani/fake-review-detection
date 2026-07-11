"""Marketplace-style entry page for ShopAI."""

from html import escape

import streamlit as st
from st_keyup import st_keyup

from utils.data import load_data
from utils.ui import marketplace_header, setup_page


setup_page("Marketplace", "AI-assisted shopping")
df = load_data()
marketplace_header()

brand, search_col, account = st.columns([1.1, 4.5, 1.2], vertical_alignment="center")
brand.markdown("## 🛍️ ShopAI")
with search_col:
    query = st_keyup(
        "Cari produk",
        placeholder="Cari produk yang ingin dibeli...",
        debounce=250,
        key="live_product_search",
        label_visibility="collapsed",
    )
account.markdown("🛒 &nbsp; 🔔 &nbsp; **Budi**")
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
if query.strip():
    catalog = catalog[catalog["product_title"].str.contains(query.strip(), case=False, na=False)]

filter_col, result_col = st.columns([1.15, 4], gap="large")
with filter_col:
    with st.container(border=True):
        st.markdown("#### 🛡️ ShopAI Trust")
        st.caption("Saring berdasarkan indikator keamanan review")
        max_risk = st.slider("Maksimum risiko", 0, 100, 100, format="%d%%")
    st.caption("Indikator AI bukan keputusan absolut mengenai keaslian review.")

catalog = catalog[catalog["risk_ratio"].le(max_risk / 100)]
with result_col:
    top_left, top_right = st.columns([2, 1])
    top_left.markdown(f"### {'Hasil pencarian' if query else 'Produk populer'}")
    top_right.caption(f"{len(catalog):,} produk ditemukan")

    if catalog.empty:
        st.warning("Produk tidak ditemukan. Coba gunakan kata kunci yang lebih singkat.")
    else:
        rows = catalog.head(24).to_dict("records")
        for start in range(0, len(rows), 3):
            columns = st.columns(3)
            for column, item in zip(columns, rows[start : start + 3]):
                with column.container(border=True):
                    st.markdown(
                        '<div style="height:165px;border-radius:10px;background:linear-gradient(135deg,#fff3ef,#ffe1d8);'
                        'display:flex;align-items:center;justify-content:center;font-size:54px">🛍️</div>',
                        unsafe_allow_html=True,
                    )
                    title = escape(str(item["product_title"]))
                    st.markdown(
                        f'<div title="{title}" style="height:72px;overflow:hidden;display:-webkit-box;'
                        f'-webkit-line-clamp:3;-webkit-box-orient:vertical;font-size:16px;font-weight:700;'
                        f'line-height:1.45;color:#172033;margin:8px 0 4px">{title}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="height:26px;color:#8a91a3;font-size:13px;display:flex;'
                        f'align-items:center">⭐ {item["rating"]:.1f} · {item["total_review"]:,} review</div>',
                        unsafe_allow_html=True,
                    )
                    score = round((1-item["risk_ratio"]) * 100)
                    st.markdown(
                        f'<div style="height:42px;box-sizing:border-box;color:#c43e24;'
                        f'padding:10px 2px;border-top:1px solid #eceef4;font-size:12px;'
                        f'font-weight:700;display:flex;align-items:center;justify-content:space-between">'
                        f'<span>🛡 AI TRUST SCORE</span><b>{score}</b></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Lihat produk", key=f"open_{item['item_id']}_{item['shop_id']}", use_container_width=True):
                        st.session_state["selected_item_id"] = str(item["item_id"])
                        st.session_state["selected_shop_id"] = str(item["shop_id"])
                        st.switch_page("pages/1_Product_Dashboard.py")
