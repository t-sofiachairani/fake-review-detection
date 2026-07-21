"""Marketplace product detail with AI-labelled customer reviews."""

from html import escape

import plotly.express as px
import pandas as pd
import streamlit as st

from utils.data import load_data
from utils.ui import (
    ORANGE,
    apply_plotly_theme,
    chart_secondary,
    icon,
    marketplace_header,
    prediction_notice,
    product_image,
    setup_page,
    trust_score_action,
    trust_score_tier,
)


setup_page("Detail Produk", show_eyebrow=False)
df = load_data()
marketplace_header()

products = (
    df[["item_id", "shop_id", "product_title"]]
    .drop_duplicates(["item_id", "shop_id"])
    .sort_values("product_title")
    .reset_index(drop=True)
)
stored_item = st.session_state.get("selected_item_id")
stored_shop = st.session_state.get("selected_shop_id")
matches = products[
    products["item_id"].eq(stored_item) & products["shop_id"].eq(stored_shop)
]
default_index = int(matches.index[0]) if not matches.empty else 0

selected_index = default_index
selected_row = products.iloc[selected_index]
product = df[
    df["item_id"].eq(selected_row["item_id"]) & df["shop_id"].eq(selected_row["shop_id"])
].copy()
title = str(selected_row["product_title"])
fake_pct = product["prediction"].eq("Fake").mean() * 100

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
with st.container(key="product_back"):
    if st.button("← Kembali", type="tertiary", use_container_width=True):
        st.switch_page("app.py")

image_col, info_col = st.columns([1, 1.7], gap="large")
with image_col:
    st.markdown(
        product_image(selected_row["item_id"], selected_row["shop_id"], detail=True),
        unsafe_allow_html=True,
    )
    st.markdown("#### Ringkasan kualitas review")
    counts = product["prediction"].value_counts().rename_axis("Status").reset_index(name="Review")
    fig = px.pie(
        counts,
        values="Review",
        names="Status",
        hole=.62,
        color="Status",
        color_discrete_map={"Fake": ORANGE, "Original": chart_secondary()},
    )
    fig.update_layout(
        height=280,
        margin=dict(l=5, r=5, t=5, b=32),
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.08),
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")
    original_col, fake_col = st.columns(2)
    original_col.markdown(
        f'<div class="stat-box"><div class="k">Original</div>'
        f'<div class="v">{100-fake_pct:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
    fake_col.markdown(
        f'<div class="stat-box"><div class="k">Fake</div>'
        f'<div class="v">{fake_pct:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Indikator pola review, bukan keputusan absolut.")
with info_col:
    st.markdown('<span class="pill">PILIHAN SHOPAI</span>', unsafe_allow_html=True)
    st.markdown(f"## {escape(title)}")
    st.markdown(f'<div style="font-size:1.35rem;font-weight:700;margin:2px 0 4px">{icon("star")} {product["rating_star"].mean():.2f} / 5</div>', unsafe_allow_html=True)
    st.caption(f"{len(product):,} review · {product['userid'].nunique():,} reviewer · Seller {selected_row['shop_id']}")
    st.divider()
    st.markdown("#### Informasi produk")
    st.write("Deskripsi, harga, stok, dan jumlah terjual belum tersedia pada dataset review.")
    c1, c2 = st.columns(2)
    trust_score = 100 - fake_pct
    score_class, score_label = trust_score_tier(trust_score)
    st.markdown(
        f'<div class="info-card trust-card {score_class}"><b>{icon("shield")} '
        f'AI Trust Score {trust_score:.0f}/100 · {score_label}</b><br>'
        f'<span style="font-size:13px">Skor dihitung dari pola keseluruhan review pada produk ini.</span></div>',
        unsafe_allow_html=True,
    )
    alert_type, moderation_message, is_held = trust_score_action(trust_score)
    if alert_type != "success":
        getattr(st, alert_type)(moderation_message)
    st.write("")
    c1, c2 = st.columns(2)
    c1.button(
        "Ditahan Moderator" if is_held else "Beli Sekarang",
        use_container_width=True,
        disabled=is_held,
        help="Pembelian dinonaktifkan selama produk ditinjau." if is_held else None,
    )
    c2.button(
        "Masukkan Keranjang",
        use_container_width=True,
        icon=":material/shopping_cart:",
        disabled=is_held,
        help="Keranjang dinonaktifkan selama produk ditinjau." if is_held else None,
    )

st.markdown("### Penilaian & review pembeli")
f1, f2, f3 = st.columns([1, 1, 2])
status = f1.selectbox("Status AI", ["Semua", "Original", "Fake"])
rating = f2.selectbox("Rating", ["Semua", 5, 4, 3, 2, 1])
search = f3.text_input("Cari di review", placeholder="Cari kata pada review produk ini...")

reviews = product.copy()
if status != "Semua":
    reviews = reviews[reviews["prediction"].eq(status)]
if rating != "Semua":
    reviews = reviews[reviews["rating_star"].eq(rating)]
if search:
    reviews = reviews[reviews["comment"].str.contains(search, case=False, na=False)]

st.caption(f"{len(reviews):,} review ditampilkan")
for row in reviews.head(100).itertuples():
    badge_class = "badge-fake" if row.prediction == "Fake" else "badge-ok"
    raw_username = getattr(row, "author_username", None)
    username = escape(str(raw_username if pd.notna(raw_username) and raw_username else f"User {row.userid}"))
    comment = escape(str(row.comment if pd.notna(row.comment) and row.comment else "Tidak ada komentar tertulis"))
    stars = "★" * int(row.rating_star) + "☆" * (5 - int(row.rating_star))
    st.markdown(
        f'<div class="review-card"><div class="review-head">'
        f'<b>{username}</b><span class="badge {badge_class}">{row.prediction}</span></div>'
        f'<div class="review-stars">{stars}</div>'
        f'<div class="review-body">{comment}</div>'
        f'<div class="review-conf">Confidence {row.confidence:.0%}</div></div>',
        unsafe_allow_html=True,
    )
prediction_notice(df)
