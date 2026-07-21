import streamlit as st

from utils.data import load_data
from utils.labels import prediction_label
from utils.ui import page_bounds, page_controls, prediction_notice, setup_page


setup_page("Review Analysis", "Review explorer")
df = load_data()
c1, c2, c3 = st.columns([1, 1, 2])
status = c1.selectbox(
    "Status AI",
    ["Semua Review", "Fake", "Original"],
    format_func=lambda value: prediction_label(value),
)
rating = c2.selectbox("Rating", ["Semua"] + [1, 2, 3, 4, 5])
query = c3.text_input("Cari review", placeholder="Ketik kata atau frasa...")

filtered = df.copy()
if status != "Semua Review": filtered = filtered[filtered["prediction"].eq(status)]
if rating != "Semua": filtered = filtered[filtered["rating_star"].eq(rating)]
if query: filtered = filtered[filtered["comment"].str.contains(query, case=False, na=False)]

st.caption(f"Menampilkan {len(filtered):,} review")
table = filtered[["comment", "rating_star", "prediction", "confidence"]].rename(columns={
    "comment": "Review", "rating_star": "Rating", "prediction": "Status AI", "confidence": "Keyakinan Model"
})
table["Status AI"] = table["Status AI"].map(prediction_label)
table["Keyakinan Model"] = table["Keyakinan Model"].map(lambda x: f"{x:.0%}")
start, end = page_bounds(len(table), key="review", page_size=20)
st.dataframe(table.iloc[start:end], use_container_width=True, hide_index=True, height=560)
page_controls(len(table), key="review", page_size=20)
prediction_notice(df)
