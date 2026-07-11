import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from utils.data import load_data
from utils.ui import setup_page


setup_page("Model Evaluation", "ML performance")
df = load_data()
label_col = "fakeornot" if "fakeornot" in df.columns else None
st.markdown("## Machine Learning Performance")
if not label_col or "model" not in set(df["prediction_source"]):
    st.warning("Evaluasi belum tersedia. Tambahkan kolom `fakeornot` dan file model `.pkl` untuk menghitung metrik terhadap ground truth.")
else:
    y_true = df[label_col].astype(str).str.lower().isin({"1", "fake", "true"}).astype(int)
    y_pred = df["prediction"].eq("Fake").astype(int)
    metrics = [accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, zero_division=0),
               recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0)]
    for col, label, value in zip(st.columns(4), ["Accuracy", "Precision", "Recall", "F1 Score"], metrics):
        col.metric(label, f"{value:.1%}")
    matrix = confusion_matrix(y_true, y_pred)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Oranges",
                    labels=dict(x="Prediction", y="Actual"), x=["Original", "Fake"], y=["Original", "Fake"])
    st.plotly_chart(fig, use_container_width=True)
st.info("Model prediction menunjukkan pola yang menyerupai fake review berdasarkan karakteristik data, bukan menentukan kepalsuan review secara absolut.")
