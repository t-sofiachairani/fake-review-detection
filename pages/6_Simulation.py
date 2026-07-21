from html import escape

import plotly.graph_objects as go
import streamlit as st

from utils.prediction import explain_review, load_meta, predict_review
from utils.ui import apply_plotly_theme, is_dark, prediction_label, setup_page


setup_page("Simulasi Review", "Coba sendiri")

EXAMPLES = {
    "Pujian generik": "mantap recommended seller josss terpercaya",
    "Repetitif spam": "bagus bagus mantap mantap recommended recommended top",
    "Original natural": "akun netflix bisa login tanpa kendala, admin responsif dan proses cepat, sudah seminggu lancar",
    "Original + keluhan": "proses agak lama sekitar sejam tapi akhirnya akun aktif dan bisa dipakai normal",
}

st.markdown("## Simulasi Deteksi Review")
st.caption(
    "Ketik atau tempel sebuah review, lalu ShopAI akan memperkirakan apakah polanya "
    "tampak wajar atau perlu ditinjau, beserta sinyal yang memengaruhinya."
)

if "sim_text" not in st.session_state:
    st.session_state["sim_text"] = EXAMPLES["Original natural"]

st.markdown("#### Contoh cepat")
example_cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(example_cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True, key=f"ex_{name}"):
        st.session_state["sim_text"] = text

review_text = st.text_area(
    "Teks review",
    key="sim_text",
    height=140,
    placeholder="Contoh: barang sesuai deskripsi, pengiriman cepat, packing rapi...",
)

analyze = st.button("Analisis Review", type="primary", use_container_width=True)

if analyze and not review_text.strip():
    st.info("Masukkan teks review terlebih dahulu.")
elif analyze:
    with st.spinner("IndoBERT sedang menganalisis review..."):
        try:
            detail = explain_review(review_text)
            fake_probability = detail["fake_probability"]
            signals = detail.get("top_signals", [])
            source = "model"
        except (ImportError, AttributeError, KeyError):
            label_fallback, confidence, source = predict_review(review_text)
            fake_probability = confidence if label_fallback == "Fake" else 1 - confidence
            signals = []
        except Exception as exc:
            st.error(f"Analisis gagal dijalankan: {exc}")
            st.stop()
    st.session_state["sim_result"] = {
        "fake_probability": float(fake_probability),
        "signals": [[str(name), float(value)] for name, value in signals],
        "source": str(source),
    }

result = st.session_state.get("sim_result")
if result:
    fake_probability = result["fake_probability"]
    signals = result["signals"]
    source = result["source"]

    label = "Fake" if fake_probability >= 0.5 else "Original"
    display_label = prediction_label(label)
    confidence = fake_probability if label == "Fake" else 1 - fake_probability
    badge_class = "badge-fake" if label == "Fake" else "badge-ok"

    result_col, gauge_col = st.columns([1, 1], gap="large")
    with result_col:
        st.markdown(
            f'<div class="review-card"><div class="review-head">'
            f'<b>Hasil Analisis</b>'
            f'<span class="badge {badge_class}">AI · {display_label}</span></div>'
            f'<div class="review-body" style="margin-top:12px;font-size:15px">'
            f'Keyakinan model: <b>{confidence:.1%}</b></div>'
            f'<div class="review-conf">Sumber: {escape(str(source))} · '
            f'probabilitas risiko {fake_probability:.1%}</div></div>',
            unsafe_allow_html=True,
        )
        if label == "Fake":
            st.warning("Pola review ini perlu ditinjau lebih lanjut.")
        else:
            st.success("Pola review ini tampak wajar menurut model.")

    with gauge_col:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=fake_probability * 100,
                number={"suffix": "%"},
                title={"text": "Probabilitas Risiko"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ee4d2d" if label == "Fake" else "#27364f"},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(39,54,79,.18)"},
                        {"range": [50, 100], "color": "rgba(238,77,45,.20)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ee4d2d", "width": 3},
                        "value": 50,
                    },
                },
            )
        )
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        apply_plotly_theme(gauge)
        st.plotly_chart(gauge, use_container_width=True)

    if signals:
        st.markdown("#### Sinyal yang paling memengaruhi")
        signal_labels = {
            "review_length": "Panjang review (kata)",
            "positive_word_ratio": "Rasio kata positif",
            "repetition_score": "Skor repetisi kata",
            "user_review_count": "Jumlah review user",
            "user_review_per_day": "Review user per hari",
            "max_cosine_similarity": "Kemiripan dengan review lain",
        }
        names = [signal_labels.get(key, key) for key, _ in signals]
        values = [round(float(value), 3) for _, value in signals]
        bar = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color="#ee4d2d" if is_dark() else "#27364f",
            )
        )
        bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        apply_plotly_theme(bar)
        st.plotly_chart(bar, use_container_width=True)
        st.caption(
            "Nilai sinyal bersifat deskriptif untuk transparansi, bukan bobot kepastian. "
            "Kombinasi seluruh fitur menentukan prediksi akhir."
        )

meta = load_meta()
if meta:
    st.divider()
    st.caption(
        f"Model aktif: `{meta.get('feature_recipe', '-')}` · "
        f"CV F1 {meta.get('candidates', {}).get(meta.get('selected_model', ''), {}).get('cv_f1_mean', 0):.1%}. "
        "Indikator AI bukan keputusan absolut mengenai keaslian review."
    )
