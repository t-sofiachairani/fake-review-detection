"""Marketplace product detail with AI-labelled customer reviews."""

from html import escape

import plotly.express as px
import pandas as pd
import streamlit as st

from utils.data import load_data
from utils.ui import ORANGE, marketplace_header, prediction_notice, setup_page


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

header_brand, _, header_account = st.columns([1.1, 4.5, 1.2], vertical_alignment="center")
header_brand.markdown("## 🛍️ ShopAI")
header_account.markdown("🛒 &nbsp; 🔔 &nbsp; **Budi**")
st.divider()
selected_index = default_index
selected_row = products.iloc[selected_index]
product = df[
    df["item_id"].eq(selected_row["item_id"]) & df["shop_id"].eq(selected_row["shop_id"])
].copy()
title = str(selected_row["product_title"])
fake_pct = product["prediction"].eq("Fake").mean() * 100

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
back_col, _ = st.columns([1, 6])
with back_col:
    if st.button("← Kembali", type="tertiary", use_container_width=True):
        st.switch_page("app.py")
st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

image_col, info_col = st.columns([1, 1.7], gap="large")
with image_col:
    st.markdown(
        '<div style="min-height:390px;border-radius:18px;background:linear-gradient(135deg,#fff3ef,#ffe1d8);'
        'display:flex;flex-direction:column;align-items:center;justify-content:center;color:#ee4d2d">'
        '<div style="font-size:90px">🛍️</div><b>Gambar produk belum tersedia</b>'
        '<span style="font-size:13px;color:#687086;margin-top:6px">Tambahkan kolom image_url untuk menampilkan gambar asli</span></div>',
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
        color_discrete_map={"Fake": ORANGE, "Original": "#27364f"},
    )
    fig.update_layout(
        height=280,
        margin=dict(l=5, r=5, t=5, b=32),
        legend=dict(orientation="h", x=.5, xanchor="center", y=-.08),
    )
    st.plotly_chart(fig, width="stretch")
    original_col, fake_col = st.columns(2)
    original_col.markdown(
        f'<div style="background:white;border:1px solid #eceef4;border-radius:12px;padding:12px">'
        f'<div style="font-size:12px;color:#687086">Original</div>'
        f'<div style="font-size:24px;font-weight:700">{100-fake_pct:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
    fake_col.markdown(
        f'<div style="background:white;border:1px solid #eceef4;border-radius:12px;padding:12px">'
        f'<div style="font-size:12px;color:#687086">Fake</div>'
        f'<div style="font-size:24px;font-weight:700">{fake_pct:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Indikator pola review, bukan keputusan absolut.")
with info_col:
    st.markdown('<span class="pill">PILIHAN SHOPAI</span>', unsafe_allow_html=True)
    st.markdown(f"## {escape(title)}")
    st.markdown(f"### ⭐ {product['rating_star'].mean():.2f} / 5")
    st.caption(f"{len(product):,} review · {product['userid'].nunique():,} reviewer · Seller {selected_row['shop_id']}")
    st.divider()
    st.markdown("#### Informasi produk")
    st.write("Deskripsi, harga, stok, dan jumlah terjual belum tersedia pada dataset review.")
    c1, c2 = st.columns(2)
    trust_score = 100 - fake_pct
    st.markdown(f'<div style="background:#eef2ff;border:1px solid #c7d2fe;padding:16px;border-radius:12px;color:#3730a3"><b>🛡 Analisis Produk oleh ShopAI &nbsp; · &nbsp; Trust Score {trust_score:.0f}/100</b><br><span style="font-size:13px">Skor dihitung dari pola keseluruhan review pada produk ini.</span></div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    c1.button("Beli Sekarang", use_container_width=True)
    c2.button("🛒 Masukkan Keranjang", use_container_width=True)

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
    badge_color = "#fff1ed" if row.prediction == "Fake" else "#edf8f2"
    text_color = "#d93d20" if row.prediction == "Fake" else "#24744b"
    raw_username = getattr(row, "author_username", None)
    username = escape(str(raw_username if pd.notna(raw_username) and raw_username else f"User {row.userid}"))
    comment = escape(str(row.comment if pd.notna(row.comment) and row.comment else "Tidak ada komentar tertulis"))
    st.markdown(
        f'<div class="panel"><div style="display:flex;justify-content:space-between;gap:12px">'
        f'<b>{username}</b><span style="background:{badge_color};color:{text_color};padding:5px 10px;'
        f'border-radius:99px;font-size:12px;font-weight:700">{row.prediction}</span></div>'
        f'<div style="color:#fbbf24;margin:7px 0">{"★" * int(row.rating_star)}{"☆" * (5-int(row.rating_star))}</div>'
        f'<div>{comment}</div><div style="color:#8a91a3;font-size:12px;margin-top:10px">Confidence {row.confidence:.0%}</div></div>',
        unsafe_allow_html=True,
    )
prediction_notice(df)
