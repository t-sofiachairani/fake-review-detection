import plotly.express as px
import streamlit as st

from utils.data import load_data
from utils.prediction import load_meta
from utils.ui import apply_plotly_theme, setup_page


setup_page("Model Evaluation", "ML performance")
load_data()
meta = load_meta()

st.markdown("## Machine Learning Performance")

if not meta:
    st.warning(
        "Metrik belum tersedia. Jalankan `python train_model.py` untuk melatih model "
        "dan menghasilkan `model/model_meta.json`."
    )
    st.stop()

final = meta["final_metrics"]
selected = meta.get("selected_model", "-")
st.caption(
    f"Model aktif: **{selected}** · dilatih {meta['trained_at'][:10]} · "
    f"{meta['n_labeled']} data berlabel (test {meta['n_test']} review)"
)

metric_row = [
    ("Accuracy", final["accuracy"]),
    ("Precision", final["precision"]),
    ("Recall", final["recall"]),
    ("F1 Score", final["f1"]),
]
if "roc_auc" in final:
    metric_row.append(("ROC-AUC", final["roc_auc"]))
for col, (label, value) in zip(st.columns(len(metric_row)), metric_row):
    col.metric(label, f"{value:.1%}")

cv_mean = meta.get("cv_accuracy_mean")
cv_std = meta.get("cv_accuracy_std")
if cv_mean is not None:
    st.caption(
        f"Baseline supervised, stratified 5-fold CV: {cv_mean:.1%} ± {cv_std:.1%}. "
        "Metrik utama di atas berasal dari student semi-supervised pada held-out test berlabel."
    )

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion Matrix (held-out test)")
    matrix = meta["confusion_matrix"]
    labels = meta["confusion_labels"]
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Oranges",
        labels=dict(x="Prediction", y="Actual", color="Reviews"),
        x=labels,
        y=labels,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=330)
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Perbandingan Kandidat Model (CV 5×3)")
    candidates = meta.get("candidates", {})
    rows = []
    for name, scores in candidates.items():
        rows.append(
            {
                "Model": name.replace("_", " ").title(),
                "CV F1": scores["cv_f1_mean"],
                "F1 ± std": scores["cv_f1_std"],
                "CV Acc": scores["cv_acc_mean"],
                "ROC-AUC": scores["cv_auc_mean"],
                "Terpilih": "✓" if name == selected else "",
            }
        )
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "CV F1": st.column_config.ProgressColumn(
                "CV F1", format="percent", min_value=0, max_value=1
            ),
            "CV Acc": st.column_config.ProgressColumn(
                "CV Acc", format="percent", min_value=0, max_value=1
            ),
            "ROC-AUC": st.column_config.ProgressColumn(
                "ROC-AUC", format="percent", min_value=0, max_value=1
            ),
            "F1 ± std": st.column_config.NumberColumn("F1 std", format="%.3f"),
        },
    )
    recipe = meta.get("feature_recipe", selected)
    st.caption(f"Recipe aktif: `{recipe}` · IndoBERT + calibrated Logistic Regression.")

pool = meta.get("unlabeled_pool", {})
eval_pseudo = meta.get("evaluation_pseudo_labels", {})
production_pseudo = meta.get("production_pseudo_labels", {})
if meta.get("training_strategy") == "balanced_teacher_agreement":
    st.markdown("#### Ringkasan Semi-Supervised")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Pool bersih", f"{pool.get('rows_after_overlap_filter', 0):,}")
    summary_cols[1].metric("Overlap dibuang", f"{pool.get('overlap_rows_removed', 0):,}")
    summary_cols[2].metric("Pseudo evaluasi", f"{eval_pseudo.get('selected_total', 0):,}")
    summary_cols[3].metric("Pseudo produksi", f"{production_pseudo.get('selected_total', 0):,}")
    st.caption(
        "Pseudo-label dipilih hanya saat calibrated Logistic Regression dan Random Forest "
        f"sepakat dengan confidence ≥ {eval_pseudo.get('teacher_confidence_threshold', 0):.0%}. "
        f"Evaluasi: {eval_pseudo.get('selected_original', 0)} original + "
        f"{eval_pseudo.get('selected_fake', 0)} fake. Produksi: "
        f"{production_pseudo.get('selected_original', 0)} original + "
        f"{production_pseudo.get('selected_fake', 0)} fake."
    )

st.markdown("#### Fitur yang digunakan model")
st.write(", ".join(f"`{feat}`" for feat in meta["features"]))
if meta.get("honest_note"):
    st.warning(meta["honest_note"])
st.info(
    "Model prediction menunjukkan pola yang menyerupai fake review berdasarkan representasi "
    "kontekstual IndoBERT dan classifier Logistic Regression terkalibrasi, bukan menentukan "
    "kepalsuan review secara absolut."
)
