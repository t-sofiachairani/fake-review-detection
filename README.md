# ShopAI — Fake Review Detection

ShopAI adalah aplikasi analitik review marketplace berbasis **Streamlit** yang membantu pengguna mengenali pola review yang menyerupai review palsu. Model menggunakan representasi bahasa Indonesia dari **IndoBERT**, dilanjutkan dengan **Logistic Regression terkalibrasi** dan strategi semi-supervised berbasis kesepakatan dua teacher.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uc12-fakereview.streamlit.app)

> Prediksi model merupakan indikator pola, bukan bukti absolut bahwa sebuah review, pengguna, atau seller melakukan kecurangan.

## Fitur aplikasi

- **Marketplace** — katalog produk dengan pencarian dan filter minimum AI Trust Score.
- **Product Dashboard** — ringkasan kualitas review, Trust Score, distribusi label, dan daftar review per produk.
- **Review Analysis** — eksplorasi review berdasarkan prediksi, rating, dan kata kunci.
- **Seller Analysis** — peringkat seller berdasarkan proporsi review yang diprediksi fake.
- **User Behavior** — aktivitas reviewer, distribusi jumlah review, frekuensi harian, dan jumlah prediksi fake.
- **Model Evaluation** — metrik holdout, confusion matrix, perbandingan kandidat, dan ringkasan pseudo-label.
- **Simulation** — analisis teks review baru menggunakan model produksi.
- **Tema terang/gelap** — tampilan konsisten untuk seluruh halaman.

## Model machine learning

Pipeline produksi:

```text
Teks review mentah
      ↓
IndoBERT mean-pooled embedding (768 dimensi)
      ↓
StandardScaler
      ↓
Calibrated Logistic Regression
      ↓
Probabilitas Fake / Original
```

### Strategi semi-supervised

Dataset memiliki 99 review dengan label asli dan 3.518 review tanpa label yang sudah bebas overlap teks. Pseudo-label hanya dipilih ketika dua teacher memberikan keputusan yang sama:

1. Calibrated IndoBERT + Logistic Regression.
2. IndoBERT + Random Forest.

Syarat pemilihan pseudo-label:

- confidence kedua teacher minimal 80%;
- kedua teacher memprediksi kelas yang sama;
- jumlah pseudo-label `Original` dan `Fake` dibuat seimbang;
- teks yang terdapat pada dataset berlabel dikeluarkan dari pool tanpa label.

Untuk evaluasi, model menggunakan 68 label training ditambah 32 pseudo-label seimbang, kemudian diuji hanya pada 31 label asli yang dikunci sejak awal. Model produksi dilatih ulang menggunakan 99 label asli ditambah 60 pseudo-label seimbang.

## Hasil evaluasi

Evaluasi dilakukan pada 31 review berlabel asli yang tidak masuk proses training maupun pseudo-labeling.

| Metrik | Nilai |
|---|---:|
| Accuracy | 93,55% |
| Precision (Fake) | 93,75% |
| Recall (Fake) | 93,75% |
| F1-score (Fake) | 93,75% |
| ROC-AUC | 98,75% |
| Brier score | 0,0578 |
| Log loss | 0,2248 |

Confusion matrix:

| Aktual | Prediksi Original | Prediksi Fake |
|---|---:|---:|
| Original | 14 | 1 |
| Fake | 1 | 15 |

Karena test set masih kecil, metrik dapat berubah bila komposisi data berubah. Brier score dan kalibrasi probabilitas digunakan agar confidence lebih informatif daripada probabilitas mentah classifier.

## AI Trust Score

AI Trust Score produk dihitung dari proporsi review pada produk tersebut yang diprediksi `Original`:

```text
AI Trust Score = (1 − proporsi prediksi Fake) × 100
```

Skor ini merangkum prediksi pada kumpulan review produk dan tidak menyatakan keaslian produk atau integritas seller secara absolut.

## Menjalankan secara lokal

### 1. Clone repository

```bash
git clone https://github.com/t-sofiachairani/fake-review-detection.git
cd fake-review-detection
```

### 2. Buat virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instal dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Streamlit

```bash
streamlit run app.py
```

Aplikasi tersedia di `http://localhost:8501`.

Artefak model dan cache embedding sudah tersedia di folder `model/`, sehingga aplikasi dapat langsung dijalankan. Analisis teks baru yang belum terdapat di cache akan memuat bobot `indobenchmark/indobert-base-p1` dari Hugging Face dan dapat lebih lambat pada pemanggilan pertama.

## Melatih ulang model

```bash
python train_model.py
```

Proses ini akan:

1. membaca dataset berlabel dan pool tanpa label;
2. menghapus overlap teks;
3. membuat atau menggunakan cache embedding IndoBERT;
4. mengevaluasi kandidat supervised;
5. memilih pseudo-label berdasarkan kesepakatan teacher;
6. mengevaluasi student pada held-out test;
7. menyimpan ulang `model/fake_review_model.pkl`, `model/embedding_cache.pkl`, dan `model/model_meta.json`.

## Struktur repository

```text
.
├── app.py                         # Halaman Marketplace / entrypoint Streamlit
├── pages/                         # Halaman multipage Streamlit
├── utils/
│   ├── data.py                    # Pembersihan dan pemuatan data
│   ├── embeddings.py              # IndoBERT dan cache embedding
│   ├── prediction.py              # Inferensi model
│   ├── semi_supervised.py         # Teacher agreement dan pseudo-labeling
│   ├── training.py                # Evaluasi kandidat model
│   └── ui.py                      # Design system dan navigasi
├── model/
│   ├── fake_review_model.pkl      # Bundle model produksi
│   ├── embedding_cache.pkl        # Cache embedding review
│   └── model_meta.json            # Metrik dan metadata training
├── data/review_shopee.csv         # Dataset review aplikasi
├── train_review_only.csv          # Dataset dengan label asli
├── train_model.py                 # Pipeline training produksi
├── tests/                         # Pengujian data, training, dan prediksi
└── requirements.txt               # Dependensi yang dipin untuk deployment
```

## Keterbatasan

- Hanya tersedia 99 review dengan label asli.
- Data didominasi review produk digital/akun, sehingga generalisasi ke domain lain belum terjamin.
- Pseudo-label bukan ground truth dan tetap dapat membawa kesalahan teacher.
- Aktivitas tinggi, teks pendek, atau kata-kata repetitif tidak otomatis membuktikan sebuah review palsu.
- Model perlu dievaluasi ulang ketika dataset, domain produk, atau distribusi bahasa berubah.

## Deployment

Aplikasi dideploy melalui Streamlit Community Cloud dari branch `main` dengan entrypoint `app.py`:

**[uc12-fakereview.streamlit.app](https://uc12-fakereview.streamlit.app)**
