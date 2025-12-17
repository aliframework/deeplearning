# 📘 Judul Proyek
*Klasifikasi Mortalitas (Kematian) Pasien Gagal Jantung Menggunakan Algoritma Machine Learning dan Deep Learning*

## 👤 Informasi
- **Nama:** Alif Rahmathul Jadid 
- **Repo:** https://github.com/aliframework/deeplearning
- **Video:** [...]  

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan klasifikasi risiko kematian akibat gagal jantung (heart failure) menggunakan dataset klinis pasien.
- Melakukan data preparation meliputi cleaning, encoding, scaling, dan Feature Selection. 
- Membangun 3 model: **Baseline**:Logistic Regresion, **Advanced**:Random Forest Classifier, **Deep Learning**:MLP  
- Melakukan evaluasi menggunakan metrik , F1-Score (Weighted) untuk menangani ketimpangan kelas (imbalanced data). Mengukur kemampuan model dalam mendeteksi pasien yang     benar-benar meninggal (menghindari false negative).

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Tingginya tingkat mortalitas pasien gagal jantung memerlukan sistem prediksi risiko yang akurat dan objektif untuk membantu tenaga medis dalam pengambilan keputusan klinis. 
- Dataset klinis bersifat tabular dan cenderung imbalanced, sehingga diperlukan teknik preprocessing dan evaluasi model yang tepat agar prediksi tidak bias terhadap kelas mayoritas.

**Goals:**  
- Membangun model klasifikasi untuk memprediksi mortalitas pasien gagal jantung dengan target akurasi minimal ≥ 80%.
- Menentukan model terbaik berdasarkan metrik evaluasi yang relevan untuk data medis (accuracy, precision, recall, F1-score, dan ROC-AUC). 

---
## 📁 Struktur Folder
```
project/
│
├── data/                   # heart_failure_clinical_records_dataset.csv
│
├── notebooks/              # Jupyter notebooks
│   └── UAS_DataScience_234311030.ipynb
│
├── src/                    # Source code
│   
├── models/                 # Saved models
│   ├── model_baseline.pkl
│   ├── model_rf.pkl
│   ├── model_heart_failure_dl.keras
│   └── scaler.pkl
├── images/                 # Visualizations
│   ├── confusion_matrix_Advanced_RF.png
│   ├── confusion_matrix_Baseline_LogReg.png
│   ├── confusion_matrix_Deep_Learning_MLP.png
│   ├── viz1_class_distribution.png
│   ├── viz2_correlation_heatmap.png
│   ├── viz3_outlier_detection.png
│   ├── viz3_training_history.png
│   └── viz5_model_comparison.png
├── Laporan Proyek Machine Learning.md
├── Checklist Submit.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber**: UCI Machine Learning Repository – Heart Failure Clinical Records
- **Jumlah Data**: 299 Baris
- **Tipe**: Tabular (Data Klinis Pasien)

### Fitur Utama
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| age | Float | Usia pasien (tahun) | 60, 75 |
| anaemia | Integer | Status anemia (0 = Tidak, 1 = Ya) | 0, 1 |
| creatinine_phosphokinase | Integer | Level CPK dalam darah | 582 |
| diabetes | Integer | Status diabetes (0 = Tidak, 1 = Ya) | 0, 1 |
| ejection_fraction | Integer | Persentase darah yang keluar dari jantung | 20, 38 |
| high_blood_pressure | Integer | Tekanan darah tinggi | 0, 1 |
| platelets | Float | Jumlah trombosit | 265000 |
| serum_creatinine | Float | Kadar kreatinin serum | 1.9 |
| serum_sodium | Integer | Kadar natrium serum | 137 |
| sex | Integer | Jenis kelamin (0 = Perempuan, 1 = Laki-laki) | 0, 1 |
| smoking | Integer | Status merokok | 0, 1 |
| time | Integer | Waktu follow-up (hari) | 130 |
| DEATH_EVENT | Integer | Target: kematian pasien | 0, 1 |

---

# 4. 🔧 Data Preparation
- **Handling Imbalance:** Menggunakan class_weight='balanced'
- **Transformasi** :
- Seluruh fitur kategorikal dalam dataset telah direpresentasikan dalam bentuk numerik biner (0 dan 1), sehingga tidak diperlukan proses encoding tambahan seperti One-Hot Encoding atau Label Encoding.
- Dilakukan proses Standardization (StandardScaler) untuk fitur numerik dengan tujuan:
* Menyamakan skala antar fitur
* Mempercepat konvergensi model
* Meningkatkan performa model berbasis jarak dan gradien
  
- **Splitting** : 
**Strategi pembagian data:**
* Training set: 80% (± 239 samples)
* Test set: 20% (± 60 samples)
* Random state: 42
---

# 5. 🤖 Modeling
- **Model 1 – Baseline:**
- - C (regularization strength): 1.0, max_iter: 100, random_state: 42
- **Model 2 – Advanced ML:** - n_estimators: 200, max_depth: 10, random_state: 42
- **Model 3 – Deep Learning:**
- 1. Input Layer: shape (input_dim)
- 2. Dense: 128 units, activation='relu'
- 3. Dropout: 0.3
- 4. Dense: 64 units, activation='relu'
- 5. Dropout: 0.3
- 6. Dense: 1 unit, activation='sigmoid'
---

# 6. 🧪 Evaluation
**Metrik:**
- **Accuracy** 
  Mengukur proporsi prediksi yang benar secara keseluruhan.
- **Recall (Sensitivity)**
  Mengukur kemampuan model dalam mendeteksi pasien yang benar-benar meninggal (menghindari false negative).
- **ROC-AUC** 
  Mengukur kemampuan model membedakan dua kelas secara keseluruhan.

### Hasil Singkat
| Model | Score (ROC-AUC) | Catatan |
|-------|-----------------|---------|
| Baseline (Logistic Regression) | 0.8652 | Sederhana dan stabil |
| Advanced (Random Forest) | 0.9037 | Performa terbaik |
| Deep Learning (MLP) | 0.8652 | Perlu data lebih besar |

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** Random Forest Classifier ditetapkan sebagai model terbaik dalam proyek prediksi mortalitas pasien gagal jantung. Model ini menunjukkan performa paling stabil dan unggul pada berbagai metrik evaluasi, termasuk Accuracy, Precision, Recall, F1-Score, dan ROC-AUC, dibandingkan dengan Logistic Regression sebagai baseline dan model Deep Learning (MLP). 
- **Alasan:**
- Keunggulan Random Forest dibandingkan model lain disebabkan oleh beberapa faktor utama:

- 1. Kemampuan Random Forest dalam menangkap hubungan non-linear dan interaksi kompleks antar fitur klinis, yang umum terjadi pada data medis.
- 2. Sifat ensemble dari Random Forest mampu mengurangi overfitting dan meningkatkan generalisasi model.
- 3. Model relatif robust terhadap noise dan outliers, yang terdapat pada beberapa fitur laboratorium seperti creatinine dan platelets.
- 4. Waktu training dan inference masih efisien dibandingkan kompleksitas model deep learning, sehingga lebih praktis untuk diterapkan. 
- **Insight penting:** 
- * Fitur klinis tertentu memiliki pengaruh besar terhadap risiko mortalitas, terutama fitur yang berkaitan dengan fungsi jantung dan ginjal seperti ejection_fraction dan serum_creatinine.
- * Dataset memiliki ketidakseimbangan kelas (imbalanced) yang berpotensi memengaruhi performa model jika tidak ditangani dengan tepat.
- * Data klinis tabular dengan ukuran relatif kecil tetap dapat memberikan hasil prediksi yang baik apabila dilakukan preprocessing dan pemilihan model yang sesuai.
---

# 8. 🔮 Future Work
- [✔] Tambah data  
- [✔] Tuning model  
- [✔] Coba arsitektur DL lain  
- [✔] Deployment  

---

# 9. 🔁 Reproducibility
Gunakan environment:
**Python 3.10+**
Libraries utama:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow` (Keras)
- `ucimlrepo`
- `matplotlib`, `seaborn`
- `joblib`

Instalasi:
```bash
pip install -r requirements.txt
