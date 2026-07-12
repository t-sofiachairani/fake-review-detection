import plotly.express as px
import streamlit as st

from utils.data import load_data
from utils.ui import ORANGE, apply_plotly_theme, chart_secondary, page_bounds, page_controls, prediction_notice, setup_page


setup_page("User Behavior", "Behavior signals")
df = load_data()
activity = df.groupby("userid", dropna=False).agg(jumlah_review=("comment", "size"),
    active_days=("ctime", lambda x: max(x.dt.date.nunique(), 1)),
    fake_prediction_count=("prediction", lambda x: x.eq("Fake").sum())).reset_index()
activity["review_per_hari"] = activity["jumlah_review"] / activity["active_days"]
activity = activity.sort_values("jumlah_review", ascending=False)

left, right = st.columns(2)
with left:
    top = activity.head(15).sort_values("jumlah_review")
    fig = px.bar(top, x="jumlah_review", y="userid", orientation="h", color_discrete_sequence=[ORANGE])
    fig.update_layout(title="User dengan review terbanyak", margin=dict(l=10, r=10, t=50, b=10))
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.histogram(activity, x="jumlah_review", nbins=25, color_discrete_sequence=[chart_secondary()])
    fig.update_layout(title="Distribusi jumlah review per user", margin=dict(l=10, r=10, t=50, b=10))
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
st.markdown("### Top Reviewer Activity")
ranked = activity[["userid", "jumlah_review", "review_per_hari", "fake_prediction_count"]]
start, end = page_bounds(len(ranked), key="behavior", page_size=20)
st.dataframe(ranked.iloc[start:end], use_container_width=True, hide_index=True)
page_controls(len(ranked), key="behavior", page_size=20)
prediction_notice(df)
