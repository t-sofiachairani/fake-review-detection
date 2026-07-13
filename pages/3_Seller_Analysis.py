import streamlit as st

from utils.data import filter_seller, load_data
from utils.ui import prediction_notice, setup_page


setup_page("Seller Analysis", "Risk overview")
df = load_data()
seller_ids = sorted(df["shop_id"].dropna().astype(str).unique())
picker_col, _ = st.columns([1.4, 3])
selected = picker_col.selectbox("Pilih seller", seller_ids)
seller = filter_seller(df, selected)
ratio = seller["prediction"].eq("Fake").mean() * 100

cols = st.columns(4)
for col, label, value in zip(cols, ["Jumlah Produk", "Jumlah Review", "Rata-rata Rating", "Review Risk Indicator"],
                             [f'{seller["item_id"].nunique():,}', f"{len(seller):,}", f'{seller["rating_star"].mean():.2f}', f"{ratio:.1f}%"]):
    col.metric(label, value)

ranking = df.groupby("shop_id", dropna=False).agg(total_review=("comment", "size"),
    fake_review=("prediction", lambda x: x.eq("Fake").sum())).reset_index()
ranking["fake_ratio"] = ranking["fake_review"] / ranking["total_review"]
ranking = ranking.sort_values(["fake_ratio", "total_review"], ascending=False)
st.markdown("### Seller Risk Ranking")
st.caption("Peringkat menunjukkan karakteristik review berisiko, bukan bukti bahwa seller melakukan fraud.")
st.dataframe(
    ranking,
    width="stretch",
    hide_index=True,
    column_config={
        "fake_ratio": st.column_config.ProgressColumn(
            "Review Risk Indicator", format="percent", min_value=0, max_value=1
        )
    },
)
prediction_notice(df)
