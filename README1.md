# Laporan Proyek Machine Learning - Ginanti Riski

## Domain Proyek

Dalam era digital saat ini, kemampuan untuk memprediksi daya beli konsumen menjadi salah satu aspek penting dalam pengambilan keputusan bisnis, khususnya di sektor otomotif. Salah satu indikator utama dalam memahami perilaku konsumen adalah **daya beli mobil**, yang dapat menggambarkan kondisi ekonomi masyarakat serta preferensi terhadap produk otomotif.

### Mengapa Masalah Ini Harus Diselesaikan?
Masalah ini perlu diselesaikan karena pemahaman yang tepat mengenai kemampuan beli pelanggan akan membantu perusahaan otomotif dan lembaga keuangan untuk:

1. **Pengambilan Keputusan Bisnis:** Perusahaan otomotif dapat mengarahkan strategi pemasaran berdasarkan segmentasi pelanggan yang lebih akurat.
2. **Efisiensi Operasional:** Lembaga keuangan dapat menyesuaikan penawaran kredit atau leasing berdasarkan kemungkinan daya beli calon konsumen.
3. **Peningkatan Customer Targeting:** Kampanye promosi bisa lebih tertarget dan mengurangi pemborosan anggaran pemasaran.

### Bagaimana Masalah Ini Diselesaikan?
Masalah ini diselesaikan dengan pendekatan *machine learning* berbasis supervisi, khususnya dengan algoritma **Random Forest**, karena kemampuannya dalam menangani data tabular dan menghindari overfitting. 
  
Sebagai referensi atas pendekatan prediksi daya beli mobil, berikut adalah salah satu studi relevan:

> [OPTIMASI HYPERPARAMETER MULTILAYER PERCEPTRON UNTUK PREDIKSI DAYA BELI MOBIL](https://e-journal.stmiklombok.ac.id/index.php/misi/article/view/739)  
> *Muhammad Iqbal, Hendri Mahmud Nawawi, M Rangga Ramadhan Saelan, Muhammad Sony Maulana, Yudhistira, Ali Mustopa*

Penelitian tersebut menyajikan pendekatan optimasi model Multilayer Perceptron (MLP) untuk memprediksi daya beli konsumen berdasarkan dataset publik dari Kaggle. Studi ini menunjukkan bahwa dengan melakukan hypertuning menggunakan algoritma Adam dan RMSprop, akurasi model dapat ditingkatkan secara signifikan hingga mencapai 92%. Temuan ini memperkuat bahwa pemilihan model dan optimasi hyperparameter yang tepat sangat krusial untuk membangun sistem prediksi yang akurat dan efektif. Penelitian ini juga menegaskan bahwa pemodelan daya beli memiliki implikasi strategis terhadap pengelolaan biaya dan perencanaan pemasaran dalam industri otomotif.

## Business Understanding

Dalam dunia otomotif, memahami pola dan potensi daya beli konsumen sangat penting untuk menentukan strategi pemasaran yang efektif. Dengan memanfaatkan data demografis seperti usia, jenis kelamin, dan pendapatan tahunan, perusahaan dapat membangun model prediktif untuk mengidentifikasi siapa saja yang berpotensi membeli mobil. Melalui proyek ini, diterapkan pendekatan machine learning untuk meningkatkan efisiensi dan akurasi pengambilan keputusan dalam proses targeting konsumen.

### Problem Statements

1. **Bagaimana cara memprediksi daya beli mobil seseorang berdasarkan informasi demografis seperti usia, jenis kelamin, pendapatan tahunan dan riwayat pembelian sebelumnya?**  
2. **Apakah model Random Forest efektif untuk melakukan klasifikasi daya beli mobil berdasarkan data yang tersedia?**  
3. **Bagaimana cara meningkatkan performa model Random Forest untuk menghindari overfitting dan menghasilkan prediksi yang optimal?**  
   
### Goals

1. **Membangun model prediksi daya beli mobil dengan input utama berupa usia, jenis kelamin, pendapatan tahunan dan riwayat pembelian sebelumnya.**  
   Model ini bertujuan untuk membantu perusahaan mengidentifikasi calon pembeli potensial secara lebih akurat.

2. **Mengevaluasi performa model Random Forest untuk menguji keefektifannya dalam mengklasifikasikan daya beli mobil konsumen.**  
   Model Random Forest dikenal tangguh dalam menangani data tabular dan fitur-fitur yang kompleks. Namun, perlu dilakukan evaluasi sejauh mana model ini efektif dalam 
   mengklasifikasikan konsumen berdasarkan data demografis yang tersedia. Evaluasi dilakukan berdasarkan metrik akurasi, precision, recall, dan f1-score, untuk mengetahui 
   seberapa efektif model dalam pengambilan keputusan bisnis.

3. **Melakukan hyperparameter tuning untuk memaksimalkan performa model Random Forest.**  
   Salah satu tantangan dalam penerapan machine learning adalah overfitting. Maka, perlu dilakukan tuning terhadap hyperparameter untuk meningkatkan generalisasi model 
   terhadap data baru. Tuning dilakukan terhadap parameter seperti `n_estimators`, `max_depth`, dan `min_samples_split` untuk menemukan konfigurasi terbaik dalam 
   meningkatkan performa prediksi.

### Solution Statements

1. **Menggunakan algoritma Random Forest sebagai baseline model.**  
   Model ini dipilih karena keandalannya dalam klasifikasi berbasis data tabular, serta kemampuannya menangani interaksi antar fitur secara otomatis.

2. **Melakukan optimasi model dengan teknik hyperparameter tuning menggunakan GridSearchCV.**  
   Ini bertujuan untuk menemukan kombinasi parameter terbaik yang menghasilkan performa optimal berdasarkan metrik akurasi dan f1-score.

3. **Mengukur performa model dengan metrik evaluasi yang terukur.**  
   Penggunaan classification report, confusion matrix, dan cross-validation dilakukan untuk memastikan model bekerja dengan baik dan mampu menggeneralisasi data baru.


## Data Understanding

Dataset yang digunakan dalam proyek ini merupakan dataset publik yang sering digunakan untuk prediksi perilaku konsumen dalam keputusan pembelian mobil. Dataset ini dapat ditemukan di berbagai repositori pembelajaran mesin, salah satunya berasal dari Kaggle, yaitu:  
🔗 [Car Purchase Prediction 🚗](https://www.kaggle.com/code/casper6290/car-purchase-prediction)

Dataset ini terdiri dari lima variabel, dengan total **1000 data**. Adapun jenis variabel dalam dataset ini adalah sebagai berikut:

### Variabel-variabel pada Dataset

| Nama Fitur     | Deskripsi                                                                                |
|----------------|-------------------------------------------------------------------------------------------|
| `User ID`      | Nomor identifikasi unik pengguna (tidak digunakan dalam pemodelan).                        |
| `Age`          | Usia pelanggan (dalam tahun). Bertipe numerik.                                             |
| `Gender`       | Jenis kelamin pelanggan (`Male`/`Female`). Bertipe kategorikal.                            |
| `AnnualSalary` | Pendapatan tahunan pelanggan (dalam USD). Bertipe numerik.                                 |
| `Purchased`    | Label target, menunjukkan apakah pelanggan membeli mobil (`1`) atau tidak (`0`).           |

---

## Exploratory Data Analysis (EDA)

Untuk memahami struktur dan karakteristik data, dilakukan beberapa tahapan eksplorasi awal berikut:

- **Informasi Dataset**  
  Mengecek jumlah baris, kolom, tipe data setiap fitur, dan memeriksa adanya nilai kosong.

- **Distribusi Variabel Target (`Purchased`)**  
  Analisis proporsi kelas pelanggan yang membeli mobil dan yang tidak. Hasil menunjukkan distribusi kelas yang relatif seimbang.

- **Pengecekan Duplikasi Data**  
  Tidak ditemukan data duplikat pada dataset.

- **Distribusi Fitur Numerik (`Age` dan `AnnualSalary`)**  
  Visualisasi menggunakan boxplot dan histogram untuk mendeteksi adanya outlier serta melihat pola sebaran nilai. Hasil menunjukkan sebaran data yang cukup wajar dan tidak terdapat outlier ekstrem.

---

### Ringkasan Hasil EDA

- Tidak terdapat nilai kosong (missing values).
- Tidak terdapat duplikasi data.
- Sebaran fitur numerik (`Age` dan `AnnualSalary`) relatif normal.
- Data siap untuk tahap preprocessing lebih lanjut seperti encoding, normalisasi, dan pembagian data.

---

## Data Preparation

Tahap ini mencakup berbagai proses pembersihan, transformasi, dan persiapan data agar siap digunakan dalam pelatihan model machine learning. Adapun langkah-langkah data preparation yang dilakukan dalam proyek ini adalah sebagai berikut:

### 1. Label Encoding
- **Proses**: Mengubah data kategorikal pada kolom `Gender` menjadi representasi numerik menggunakan teknik label encoding.
- **Alasan**: Algoritma machine learning memerlukan input dalam format numerik untuk dapat diproses secara optimal.

### 2. Cek Korelasi Antar Variabel
- **Proses**: Menghitung korelasi antar fitur numerik dan label target `Purchased`, serta divisualisasikan menggunakan heatmap.
- **Alasan**: Untuk memahami hubungan antar variabel, memilih fitur yang relevan, dan mendeteksi kemungkinan multikolinearitas.

### 3. Penghapusan Variabel Tidak Penting
- **Proses**: Menghapus kolom `User ID` dari dataset.
- **Alasan**: Kolom ini bersifat unik untuk masing-masing data, sehingga tidak berkontribusi terhadap proses prediksi dan dapat dihapus untuk mengurangi kompleksitas data.

### 4. Normalisasi Data
- **Proses**: Melakukan normalisasi pada fitur numerik seperti `Age` dan `AnnualSalary` menggunakan metode standardisasi (mean = 0, standar deviasi = 1).
- **Alasan**: Walaupun Random Forest tidak membutuhkan normalisasi, tahap ini disiapkan untuk fleksibilitas penggunaan algoritma lain di masa depan.

### 5. Pemisahan Fitur dan Target
- **Proses**: Memisahkan dataset menjadi fitur (`X`) dan target (`y`).
- **Alasan**: Agar fitur input dan label output dapat diproses secara terpisah dalam tahap pelatihan model.

### 6. Oversampling dengan SMOTE
- **Proses**: Menyeimbangkan jumlah data pada masing-masing kelas target menggunakan metode Synthetic Minority Oversampling Technique (SMOTE).
- **Alasan**: Untuk mengatasi ketidakseimbangan kelas yang dapat menyebabkan bias model terhadap kelas mayoritas.

### 7. Data Splitting
- **Proses**: Membagi data menjadi data latih (training set) dan data uji (testing set) dengan rasio 80:20.
- **Alasan**: Untuk mengukur performa model pada data yang tidak pernah dilihat sebelumnya, sehingga dapat menguji kemampuan generalisasi model.

---

## Modeling

Pada tahap ini, dilakukan proses pemodelan untuk memprediksi apakah seorang pelanggan akan membeli mobil atau tidak berdasarkan fitur-fitur seperti usia, gender, dan pendapatan tahunan.

### Algoritma yang Digunakan

Model yang digunakan dalam proyek ini adalah **Random Forest Classifier**, yaitu ensemble learning method berbasis decision tree yang bekerja dengan membangun beberapa decision tree dan menggabungkannya untuk menghasilkan prediksi yang lebih stabil dan akurat.

#### Kelebihan Random Forest
- Tahan terhadap overfitting, terutama pada dataset besar dan kompleks.
- Dapat menangani fitur kategorikal dan numerik secara bersamaan.
- Mampu memberikan estimasi pentingnya fitur.
- Hasil yang lebih stabil dibanding decision tree tunggal.

#### Kekurangan Random Forest
- Kurang efisien pada data real-time karena kompleksitas tinggi.
- Model sulit untuk diinterpretasikan dibanding decision tree biasa.

### Baseline Model

Baseline model dibuat menggunakan parameter default dari `RandomForestClassifier`:

```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
```

### Model Improvement: Hyperparameter Tuning
Untuk meningkatkan performa baseline, dilakukan tuning hyperparameter menggunakan GridSearchCV dari ```python sklearn.model_selection ```.

Parameter grid yang diuji:
```python
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```
Grid search dilakukan dengan 5-fold cross-validation:

```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
```

Hasil Tuning
Hasil dari GridSearchCV menunjukkan parameter terbaik sebagai berikut:

```python
Best Parameters: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
```

Model terbaik kemudian digunakan untuk prediksi pada data testing dan dievaluasi dengan:

```python
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
```

Model akhir dipilih karena memberikan keseimbangan terbaik antara akurasi dan kompleksitas serta menunjukkan generalisasi yang baik terhadap data baru.

## Evaluation

Pada tahap evaluasi, digunakan beberapa metrik evaluasi klasifikasi untuk menilai performa model. Karena proyek ini merupakan permasalahan klasifikasi biner (membeli mobil atau tidak), maka diperlukan lebih dari sekadar akurasi untuk menilai performa model secara menyeluruh. Metrik yang digunakan adalah:

### Confusion Matrix

Confusion matrix adalah representasi visual dari hasil prediksi model klasifikasi, yang memperlihatkan jumlah prediksi benar dan salah untuk masing-masing kelas.

|                | Predicted Positive (1) | Predicted Negative (0) |
|----------------|------------------------|-------------------------|
| Actual Positive (1)   | True Positive (TP)       | False Negative (FN)        |
| Actual Negative (0)   | False Positive (FP)      | True Negative (TN)         |

**Penjelasan:**
- **True Positive (TP)**: Prediksi benar untuk kelas positif.
- **True Negative (TN)**: Prediksi benar untuk kelas negatif.
- **False Positive (FP)**: Prediksi positif padahal sebenarnya negatif (type I error).
- **False Negative (FN)**: Prediksi negatif padahal sebenarnya positif (type II error).

**Rumus dari metrik evaluasi berdasarkan confusion matrix:**

## Rumus dari Metrik Evaluasi Berdasarkan Confusion Matrix

- **Accuracy**  
  Accuracy = (TP + TN)/(TP + TN + FP + FN)

- **Precision**  
  Precision = TP/(TP + FP)

- **Recall (Sensitivity)**  
  Recall = TP/(TP + FN)

- **F1-score**  
  F1-score = (2 x Precision x Recall)/(Precision + Recall)

Metrik ini sangat penting karena proyek ini merupakan problem klasifikasi biner (**membeli mobil** atau **tidak**), sehingga tidak cukup hanya mengandalkan akurasi saja. Evaluasi menyeluruh dengan precision, recall, dan F1-score memberikan gambaran yang lebih akurat terhadap kinerja model.

---

### Hasil Evaluasi Sebelum Hyperparameter Tuning

| Metrik     | Kelas 0 | Kelas 1 | Macro Avg | Weighted Avg |
|------------|---------|---------|-----------|--------------|
| Precision  | 0.92    | 0.93    | 0.93      | 0.93         |
| Recall     | 0.94    | 0.91    | 0.92      | 0.93         |
| F1-score   | 0.93    | 0.92    | 0.92      | 0.92         |
| Accuracy   | -       | -       | -         | **92.50%**   |

**Confusion Matrix:**

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| Actual 0      |     121     |      8      |
| Actual 1      |     10      |     101     |

---

### Hasil Evaluasi Setelah Hyperparameter Tuning

Tuning dilakukan menggunakan `GridSearchCV` dengan 384 kombinasi parameter dan 5-fold cross-validation. Model terbaik didapatkan dengan parameter:

```python
{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}
```
| Metrik     | Kelas 0 | Kelas 1 | Macro Avg | Weighted Avg |
|------------|---------|---------|-----------|--------------|
| Precision  | 0.93    | 0.94    | 0.93      | 0.93         |
| Recall     | 0.95    | 0.92    | 0.93      | 0.93         |
| F1-score   | 0.94    | 0.93    | 0.93      | 0.93         |
| Accuracy   | -       | -       | -         | **93.33%**   |

**Confusion Matrix:**

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| Actual 0      |     1212    |      7      |
| Actual 1      |     9      |     102     |

## Kesimpulan Evaluasi

Model yang telah melalui proses tuning menunjukkan peningkatan akurasi dari **92.50%** menjadi **93.33%**. Selain itu, nPeningkatan juga terlihat pada metrik precision, recall, dan F1-score, yang menandakan model lebih seimbang dalam mendeteksi dua kelas (**membeli** dan **tidak membeli mobil**). Hal ini penting untuk menjaga kualitas prediksi dalam konteks pemasaran dan targeting pelanggan.

Dengan demikian, **hyperparameter tuning** terbukti efektif dalam meningkatkan performa model **Random Forest** pada kasus klasifikasi daya beli pelanggan ini.


