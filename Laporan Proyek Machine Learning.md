## INFORMASI PROYEK

**Judul Proyek:**  
[Klasifikasi Mortalitas (Kematian) Pasien Gagal Jantung Menggunakan Algoritma Machine Learning dan Deep Learning]

**Nama Mahasiswa:** [Alif Rahmathul]  
**NIM:** [234311030]  
**Program Studi:** [Teknologi Rekayasa Perangkat Lunak]  
**Mata Kuliah:** [Data Science]  
**Dosen Pengampu:** [Gus Nanang Syaifuddin,S.Kom.,M.Kom.]  
**Tahun Akademik:** [2025/5]
**Link GitHub Repository:** [URL Repository]
**Link Video Pembahasan:** [URL Repository]

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. Memahami konteks masalah dan merumuskan problem statement secara jelas
2. Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. Melakukan data preparation yang sesuai dengan karakteristik dataset
4. Mengembangkan tiga model machine learning yang terdiri dari (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML
6. Melaporkan hasil eksperimen secara ilmiah dan sistematis
7. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
8. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW

### 2.1 Latar Belakang
Mengapa Proyek Ini Penting?
Proyek ini penting karena Gagal Jantung (Heart Failure/HF) adalah masalah kesehatan global dengan tingkat mortalitas yang tinggi. Kemampuan untuk secara akurat dan cepat mengidentifikasi pasien berisiko tinggi adalah krusial untuk intervensi medis yang tepat waktu dan terpersonalisasi. Model prediksi berbasis data dapat meningkatkan objektivitas dalam penilaian risiko dibandingkan metode tradisional.

Permasalahan Umum pada Domain Terkait (Kesehatan)
Permasalahan umum terletak pada kompleksitas diagnosis dan prognosis HF. Penilaian risiko secara tradisional (berdasarkan skor klinis atau rule of thumb) seringkali memiliki keterbatasan karena interaksi antar variabel klinis (seperti usia, fraksi ejeksi, dan kadar serum) bersifat rumit, non-linear, dan sulit ditangkap oleh model statistik sederhana.

Manfaat Proyek untuk Pengguna, Bisnis, atau Penelitian

* Pengguna (Tenaga Medis): Menyediakan Sistem Pendukung Keputusan Klinis (Clinical Decision Support System—CDSS) yang objektif untuk membantu mengkategorikan pasien berisiko tinggi.

* Penelitian: Menunjukkan efektivitas komparatif antara model Machine Learning dan Deep Learning dalam mengekstrak pola kompleks dari data klinis, menjadi dasar untuk studi prognosis penyakit kardiovaskular lanjutan.

* Manfaat Utama: Berpotensi meningkatkan kualitas perawatan pasien dan survival rate dengan memungkinkan intervensi medis yang lebih dini dan terfokus.

Studi Literatur atau Referensi Ilmiah
Referensi yang digunakan dalam proyek ini adalah:

1. Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction only. BMC Medical Informatics and Decision Making, 20(1).

2. Al-Hussainy, G. P. M. F. C. A., Al-Musawi, A. H., & Al-Tameemi, A. R. R. (2022). Deep learning-based predictive models for cardiovascular disease: a systematic review. Expert Systems with Applications, 208, 118182.

**Contoh referensi (berformat APA/IEEE):**
> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

| Aspek         | Deskripsi                                                                 |
|:--------------|:--------------------------------------------------------------------------|
| Domain        | Kesehatan (Kardiologi)                                                    |
| Kondisi       | Gagal Jantung (Heart Failure/HF), penyebab morbiditas dan mortalitas signifikan |
| Tujuan Medis  | Menentukan risiko mortalitas pasien dalam jangka waktu tertentu (prognosis) |
| Data          | Data klinis pasien selama follow-up: variabel demografi, kebiasaan, dan tes laboratorium (contoh: creatinine_serum, ejection_fraction, platelets) |
| Target Output | Variabel biner DEATH_EVENT (0: Bertahan Hidup, 1: Meninggal)              |

### Latar Belakang Penerapan Data Science

Secara historis, dokter mengandalkan skor risiko klinis atau penilaian subjektif. Metode ini seringkali gagal menangkap interaksi kompleks antar variabel. Contohnya, hubungan antara kadar $creatinine\_serum$ yang tinggi dan $ejection\_fraction$ yang rendah mungkin memiliki efek non-linear terhadap risiko kematian yang sulit diukur oleh model statistik linear sederhana.

#### Justifikasi Machine Learning (ML/DL)
- **Identifikasi Pola Kompleks**: ML/DL unggul dalam menemukan pola non-linear dan interaksi kompleks antar fitur (variabel klinis) yang tersembunyi.
- **Objektivitas**: Model ML menawarkan alat bantu keputusan yang objektif, mengurangi variasi dalam penilaian risiko antar dokter.
- **Peningkatan Akurasi**: Penelitian (seperti yang dilakukan oleh Chicco & Jurman, 2020) menunjukkan bahwa model ML dapat melebihi performa skor klinis tradisional, bahkan dengan jumlah fitur yang terbatas.

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING
### 3.1 Problem Statements
Tuliskan 2–4 pernyataan masalah yang jelas dan spesifik.

**Contoh (universal):**
1. Model perlu mampu memprediksi nilai target dengan akurasi tinggi
2. Sistem harus dapat mengidentifikasi pola pada citra secara otomatis
3. Dataset memiliki noise sehingga perlu preprocessing yang tepat
4. Dibutuhkan model deep learning yang mampu belajar representasi fitur kompleks

Berdasarkan latar belakang dan karakteristik domain kesehatan kardiovaskular, permasalahan yang diangkat dalam proyek ini adalah sebagai berikut:

1. Tingginya tingkat mortalitas pasien gagal jantung memerlukan sistem prediksi risiko yang akurat dan objektif untuk membantu tenaga medis dalam pengambilan keputusan klinis.

2. Pendekatan tradisional berbasis skor klinis sering kali tidak mampu menangkap hubungan non-linear dan interaksi kompleks antar variabel klinis seperti usia, fraksi ejeksi, dan kadar kreatinin serum.

3. Dataset klinis bersifat tabular dan cenderung imbalanced, sehingga diperlukan teknik preprocessing dan evaluasi model yang tepat agar prediksi tidak bias terhadap kelas mayoritas.

4. Perlu dilakukan perbandingan performa antara model baseline, machine learning lanjutan, dan deep learning untuk menentukan pendekatan terbaik dalam memprediksi mortalitas pasien gagal jantung.
### 3.2 Goals

Tujuan harus spesifik, terukur, dan selaras dengan problem statement.
**Contoh tujuan:**
1. Membangun model ML untuk memprediksi variabel target dengan akurasi > 80%
2. Mengukur performa tiga pendekatan model (baseline, advanced, deep learning)
3. Menentukan model terbaik berdasarkan metrik evaluasi yang relevan
4. Menghasilkan sistem yang dapat bekerja secara reproducible


Tujuan dari proyek data science ini adalah:

1. Membangun model klasifikasi untuk memprediksi mortalitas pasien gagal jantung dengan target akurasi minimal ≥ 80%.

2. Membandingkan performa tiga pendekatan model, yaitu baseline model, advanced machine learning model, dan deep learning model.

3. Menentukan model terbaik berdasarkan metrik evaluasi yang relevan untuk data medis (accuracy, precision, recall, F1-score, dan ROC-AUC).

4. Menghasilkan pipeline analisis yang reproducible, terdokumentasi dengan baik, dan dapat digunakan sebagai dasar sistem pendukung keputusan klinis.

### 3.3 Solution Approach

Mahasiswa **WAJIB** menggunakan minimal **tiga model** dengan komposisi sebagai berikut:
#### **Model 1 – Baseline Model**
Model sederhana sebagai pembanding dasar.
**Pilihan model:**
- Linear Regression (untuk regresi)
- Logistic Regression (untuk klasifikasi)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

Model yang dipilih: Logistic Regression

Alasan Pemilihan:

Logistic Regression merupakan model klasifikasi linier yang sederhana dan umum digunakan dalam bidang medis.

Model ini mudah diinterpretasikan dan sering dijadikan baseline dalam penelitian prediksi klinis.

Memberikan pembanding dasar sebelum menggunakan model yang lebih kompleks.

Logistic Regression memodelkan probabilitas kejadian kematian (DEATH_EVENT = 1) berdasarkan kombinasi linier dari fitur-fitur klinis yang tersedia.
#### **Model 2 – Advanced / ML Model**
Model machine learning yang lebih kompleks.
**Pilihan model:**
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Support Vector Machine (SVM)
- Ensemble methods
- Clustering (K-Means, DBSCAN) - untuk unsupervised
- PCA / dimensionality reduction (untuk preprocessing)

Model yang dipilih: Random Forest Classifier

Alasan Pemilihan:

* Random Forest mampu menangkap hubungan non-linear antar fitur.

* Lebih robust terhadap noise dan outliers dibanding model linier.

* Menyediakan informasi feature importance yang berguna untuk interpretasi medis.

Random Forest merupakan ensemble dari beberapa decision tree yang bekerja secara voting, sehingga meningkatkan stabilitas dan akurasi prediksi.
#### **Model 3 – Deep Learning Model (WAJIB)**
Model deep learning yang sesuai dengan jenis data.
**Pilihan Implementasi (pilih salah satu sesuai dataset):**
**A. Tabular Data:**
- Multilayer Perceptron (MLP) / Neural Network
- Minimum: 2 hidden layers
- Contoh: prediksi harga, klasifikasi binary/multiclass

**B. Image Data:**
- CNN sederhana (minimum 2 convolutional layers) **ATAU**
- Transfer Learning (ResNet, VGG, MobileNet, EfficientNet) - **recommended**
- Contoh: klasifikasi gambar, object detection

**C. Text Data:**
- LSTM/GRU (minimum 1 layer) **ATAU**
- Embedding + Dense layers **ATAU**
- Pre-trained model (BERT, DistilBERT, Word2Vec)
- Contoh: sentiment analysis, text classification

**D. Time Series:**
- LSTM/GRU untuk sequential prediction
- Contoh: forecasting, anomaly detection

**E. Recommender Systems:**
- Neural Collaborative Filtering (NCF)
- Autoencoder-based Collaborative Filtering
- Deep Matrix Factorization

**Minimum Requirements untuk Deep Learning:**
- ✅ Model harus training minimal 10 epochs
- ✅ Harus ada plot loss dan accuracy/metric per epoch
- ✅ Harus ada hasil prediksi pada test set
- ✅ Training time dicatat (untuk dokumentasi)

**Tidak Diperbolehkan:**
- ❌ Copy-paste kode tanpa pemahaman
- ❌ Model tidak di-train (hanya define arsitektur)
- ❌ Tidak ada evaluasi pada test set

Model yang dipilih: Multilayer Perceptron (MLP)

Jenis Deep Learning:
Multilayer Perceptron (MLP) – untuk data tabular

Alasan Pemilihan:

- Dataset berbentuk data tabular numerik, sehingga MLP merupakan arsitektur deep learning yang paling sesuai.

- MLP mampu mempelajari representasi non-linear yang kompleks dari data klinis.

- Digunakan untuk membandingkan efektivitas deep learning terhadap model ML konvensional pada dataset berukuran kecil-menengah.

---

## 4. DATA UNDERSTANDING
### 4.1 Informasi Dataset
**Sumber Dataset:**  
UCI Machine Learning Repository / Kaggle
Dataset: Heart Failure Clinical Records Dataset
URL: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

**Deskripsi Dataset:**
- Jumlah baris (rows): 299
- Jumlah kolom (columns/features): 13
- Tipe data: Tabular
- Ukuran dataset: ± 0.05 MB
- Format file: CSV

### 4.2 Deskripsi Fitur
Jelaskan setiap fitur/kolom yang ada dalam dataset.
**Contoh tabel:**
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| id | Integer | ID unik data | 1, 2, 3 |
| age | Integer | Usia (tahun) | 25, 30, 45 |
| income | Float | Pendapatan (juta) | 5.5, 10.2 |
| category | Categorical | Kategori produk | A, B, C |
| text | String | Teks ulasan | "Produk bagus..." |
| image | Image | Citra 224x224 RGB | Array 224x224x3 |
| label | Categorical | Label target | 0, 1 atau "positif", "negatif" |

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

### 4.3 Kondisi Data

Jelaskan kondisi dan permasalahan data:

- **Missing Values:** Tidak ditemukan missing values.
- **Duplicate Data:** Tidak ditemukan data duplicate
- **Outliers:** Terdapat outliers pada fitur `creatinine_phosphokinase`, `platelets`, dan `serum_creatinine`.
- **Imbalanced Data:** Ya
* DEATH_EVENT = 0 : ± 68%
* DEATH_EVENT = 1 : ± 32%
- **Noise:** Ada potensi noise pada fitur laboratorium dengan rentang nilai yang sangat besar.
- **Data Quality Issues:** Dataset relatif bersih namun berukuran kecil.

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

**Requirement:** Minimal 3 visualisasi yang bermakna dan insight-nya.
**Contoh jenis visualisasi yang dapat digunakan:**
- Histogram (distribusi data)
- Boxplot (deteksi outliers)
- Heatmap korelasi (hubungan antar fitur)
- Bar plot (distribusi kategori)
- Scatter plot (hubungan 2 variabel)
- Wordcloud (untuk text data)
- Sample images (untuk image data)
- Time series plot (untuk temporal data)
- Confusion matrix heatmap
- Class distribution plot


#### Visualisasi 1: [Judul Visualisasi]
[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 2: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 3: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]



---

## 5. DATA PREPARATION

Bagian ini menjelaskan **semua** proses transformasi dan preprocessing data yang dilakukan.
### 5.1 Data Cleaning
**Aktivitas:**
- Handling missing values
- Removing duplicates
- Handling outliers
- Data type conversion
**Contoh:**
```
Missing Values:
- Fitur 'age' memiliki 50 missing values (5% dari data)
- Strategi: Imputasi dengan median karena distribusi skewed
- Alasan: Median lebih robust terhadap outliers dibanding mean
```

**[Jelaskan langkah-langkah data cleaning yang Anda lakukan]**



### 5.2 Feature Engineering
**Aktivitas:**
- Creating new features
- Feature extraction
- Feature selection
- Dimensionality reduction

**[Jelaskan feature engineering yang Anda lakukan]**

### 5.3 Data Transformation

**Untuk Data Tabular:**
- Encoding (Label Encoding, One-Hot Encoding, Ordinal Encoding)
- Scaling (Standardization, Normalization, MinMaxScaler)

**Untuk Data Text:**
- Tokenization
- Lowercasing
- Removing punctuation/stopwords
- Stemming/Lemmatization
- Padding sequences
- Word embedding (Word2Vec, GloVe, fastText)

**Untuk Data Image:**
- Resizing
- Normalization (pixel values 0-1 atau -1 to 1)
- Data augmentation (rotation, flip, zoom, brightness, etc.)
- Color space conversion

**Untuk Time Series:**
- Creating time windows
- Lag features
- Rolling statistics
- Differencing

**[Jelaskan transformasi yang Anda lakukan]**

### 5.4 Data Splitting

**Strategi pembagian data:**
```
- Training set: [X]% ([jumlah] samples)
- Validation set: [X]% ([jumlah] samples) - jika ada
- Test set: [X]% ([jumlah] samples)
```
**Contoh:**
```
Menggunakan stratified split untuk mempertahankan distribusi kelas:
- Training: 80% (8000 samples)
- Test: 20% (2000 samples)
- Random state: 42 untuk reproducibility
```

**[Jelaskan strategi splitting Anda dan alasannya]**



### 5.5 Data Balancing (jika diperlukan)
**Teknik yang digunakan:**
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Undersampling
- Class weights
- Ensemble sampling

**[Jelaskan jika Anda melakukan data balancing]**

### 5.6 Ringkasan Data Preparation

**Per langkah, jelaskan:**
1. **Apa** yang dilakukan
**[Jelaskan ]**
2. **Mengapa** penting
**[Jelaskan Mengapa ?]**
3. **Bagaimana** implementasinya
**[Jelaskan Bagaimana]**

---

## 6. MODELING
### 6.1 Model 1 — Baseline Model
#### 6.1.1 Deskripsi Model

**Nama Model:** [Nama model, misal: Logistic Regression]
**Teori Singkat:**  
[Jelaskan secara singkat bagaimana model ini bekerja]
**Alasan Pemilihan:**  
[Mengapa memilih model ini sebagai baseline?]

#### 6.1.2 Hyperparameter
**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- C (regularization): 1.0
- solver: 'lbfgs'
- max_iter: 100
```

#### 6.1.3 Implementasi (Ringkas)
```python
# Contoh kode (opsional, bisa dipindah ke GitHub)
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=1.0, max_iter=100)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Hasil Awal

**[Tuliskan hasil evaluasi awal, akan dijelaskan detail di Section 7]**

---

### 6.2 Model 2 — ML / Advanced Model
#### 6.2.1 Deskripsi Model

**Nama Model:** [Nama model, misal: Random Forest / XGBoost]
**Teori Singkat:**  
[Jelaskan bagaimana algoritma ini bekerja]

**Alasan Pemilihan:**  
[Mengapa memilih model ini?]

**Keunggulan:**
- [Sebutkan keunggulan]

**Kelemahan:**
- [Sebutkan kelemahan]

#### 6.2.2 Hyperparameter

**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- n_estimators: 100
- max_depth: 10
- learning_rate: 0.1
- min_samples_split: 2
```

**Hyperparameter Tuning (jika dilakukan):**
- Metode: [Grid Search / Random Search / Bayesian Optimization]
- Best parameters: [...]

#### 6.2.3 Implementasi (Ringkas)
```python
# Contoh kode
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Hasil Model

**[Tuliskan hasil evaluasi, akan dijelaskan detail di Section 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model

**Nama Model:** [Nama arsitektur, misal: CNN / LSTM / MLP]

** (Centang) Jenis Deep Learning: **
- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Mengapa arsitektur ini cocok untuk dataset Anda?]

#### 6.3.2 Arsitektur Model

**Deskripsi Layer:**

[Jelaskan arsitektur secara detail atau buat tabel]

**Contoh:**
```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [jumlah]
Trainable parameters: [jumlah]
```

#### 6.3.3 Input & Preprocessing Khusus

**Input shape:** [Sebutkan dimensi input]  
**Preprocessing khusus untuk DL:**
- [Sebutkan preprocessing khusus seperti normalisasi, augmentasi, dll.]

#### 6.3.4 Hyperparameter

**Training Configuration:**
```
- Optimizer: Adam / SGD / RMSprop
- Learning rate: [nilai]
- Loss function: [categorical_crossentropy / mse / binary_crossentropy / etc.]
- Metrics: [accuracy / mae / etc.]
- Batch size: [nilai]
- Epochs: [nilai]
- Validation split: [nilai] atau menggunakan validation set terpisah
- Callbacks: [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, etc.]
```

#### 6.3.5 Implementasi (Ringkas)

**Framework:** TensorFlow/Keras / PyTorch
```python
# Contoh kode TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_dl.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
```

#### 6.3.6 Training Process

**Training Time:**  
[Sebutkan waktu training total, misal: 15 menit]

**Computational Resource:**  
[CPU / GPU, platform: Local / Google Colab / Kaggle]

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? [Ya/Tidak, jelaskan]
- Apakah model sudah converge? [Ya/Tidak, jelaskan]
- Apakah perlu lebih banyak epoch? [Ya/Tidak, jelaskan]

#### 6.3.7 Model Summary
```
[Paste model.summary() output atau rangkuman arsitektur]
```

---

## 7. EVALUATION

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**
- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**
- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**
- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**[Pilih dan jelaskan metrik yang Anda gunakan]**

### 7.2 Hasil Evaluasi Model

#### 7.2.1 Model 1 (Baseline)

**Metrik:**
```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Opsional: tampilkan beberapa contoh prediksi]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Baseline (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Advanced (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| Deep Learning (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting:**  
   [Analisis apakah model mengalami overfitting atau underfitting]

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.2 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**
- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**
- [Insight 1]
- [Insight 2]

### 8.3 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll

```
