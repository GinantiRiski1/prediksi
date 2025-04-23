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
Dataset yang digunakan dalam proyek ini merupakan dataset publik yang umumnya digunakan untuk prediksi perilaku konsumen terhadap keputusan pembelian mobil. Dataset ini dapat ditemukan di berbagai repositori pembelajaran mesin, salah satunya berasal dari Kaggle. yaitu: [Car Purchase Prediction 🚗](https://www.kaggle.com/code/casper6290/car-purchase-prediction).

Dataset ini terdiri dari lima variabel, yaitu:

### Variabel-variabel pada dataset:
- **User ID**: Nomor identifikasi unik pengguna (tidak digunakan dalam pemodelan).
- **Age**: Usia pelanggan (dalam tahun).
- **Gender**: Jenis kelamin pelanggan (Male/Female).
- **AnnualSalary**: Pendapatan tahunan pelanggan (dalam USD).
- **Purchased**: Label target yang menunjukkan apakah pelanggan membeli mobil (`1`) atau tidak (`0`).


### Exploratory Data Analysis (EDA)
Untuk memahami struktur dan distribusi data, dilakukan beberapa tahapan eksplorasi data, di antaranya:

- **Distribusi variabel target (`Purchased`)**: Analisis dilakukan untuk melihat keseimbangan data antara kelas yang membeli mobil dan yang tidak.
- **Distribusi fitur numerik**: Visualisasi dilakukan untuk memeriksa sebaran nilai pada fitur `Age` dan `AnnualSalary`, guna mendeteksi outlier dan pola distribusi.
- **Korelasi antar variabel numerik**: Korelasi antara `Age`, `AnnualSalary`, dan label `Purchased` dihitung untuk mengetahui hubungan linier antar variabel yang dapat mendukung prediksi.

Hasil EDA menunjukkan bahwa fitur numerik memiliki distribusi yang cukup normal dan tidak terdapat nilai hilang. Oleh karena itu, dataset ini siap digunakan dalam tahap pemodelan supervised learning.

## Data Preparation

Tahapan ini mencakup berbagai proses pembersihan dan transformasi data agar siap digunakan dalam proses pelatihan model machine learning. Adapun langkah-langkah data preparation yang dilakukan dalam proyek ini adalah sebagai berikut:

### 1. Cek Jumlah dan Informasi Data
- **Proses**: Meninjau jumlah baris dan kolom dalam dataset.
- **Alasan**: Untuk mengetahui struktur data awal, tipe data setiap kolom, dan mendeteksi nilai null.

### 2. Cek Distribusi Data
- **Proses**: Menampilkan proporsi kelas dari target `Purchased`.
- **Alasan**: Untuk mengetahui apakah kelas target seimbang atau perlu penanganan lebih lanjut.

### 3. Cek Missing Values
- **Proses**: Mengecek nilai kosong pada setiap kolom.
- **Alasan**: Nilai kosong dapat mengganggu pelatihan model, sehingga harus ditangani.

### 4. Cek Duplikasi Data
- **Proses**: Mendeteksi dan menghapus baris duplikat.
- **Alasan**: Duplikasi dapat menyebabkan bias dan menurunkan performa model.

### 5. Cek Outliers Data
- **Proses**: Melihat nilai ekstrem pada fitur numerik seperti `Age` dan `AnnualSalary`.
- **Alasan**: Outlier dapat mempengaruhi distribusi data dan mengganggu proses pelatihan.

### 6. Label Encoding
- **Proses**: Mengubah data kategorikal (`Gender`) menjadi numerik.
- **Alasan**: Algoritma machine learning memerlukan input numerik untuk dapat diproses.

### 7. Delete Variabel Tidak Penting
- **Proses**: Menghapus kolom `User ID`.
- **Alasan**: Kolom ini bersifat unik dan tidak memberikan kontribusi terhadap prediksi.

### 8. Cek Korelasi Antar Variabel
- **Proses**: Menggunakan heatmap korelasi untuk melihat hubungan antar fitur dan target.
- **Alasan**: Untuk mengevaluasi relevansi fitur terhadap target dan menghindari multikolinearitas.

### 9. Normalisasi Data
- **Proses**: Melakukan normalisasi pada fitur numerik.
- **Alasan**: Meski Random Forest tidak memerlukannya, normalisasi disiapkan agar model lain bisa digunakan bila diperlukan.

### 10. Pisahkan Fitur dan Target
- **Proses**: Memisahkan `X` (fitur) dan `y` (target) sebelum proses pelatihan.
- **Alasan**: Supaya proses training model bisa dilakukan dengan benar.

### 11. Oversampling Data
- **Proses**: Menyeimbangkan jumlah kelas target dengan teknik oversampling.
- **Alasan**: Untuk mencegah model bias terhadap kelas mayoritas.

### 12. Data Splitting
- **Proses**: Membagi data menjadi data latih dan data uji (80:20).
- **Alasan**: Untuk mengevaluasi performa model terhadap data yang tidak dilatih.


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


