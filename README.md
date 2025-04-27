# Laporan Proyek Machine Learning - Ginanti Riski

## Project Overview

Sistem rekomendasi film merupakan salah satu aplikasi Machine Learning yang sangat populer, banyak digunakan oleh platform streaming seperti Netflix, Hulu, dan Amazon Prime Video. Dengan terus bertambahnya jumlah film dan serial, pengguna sering kali mengalami kesulitan dalam menemukan tontonan yang sesuai dengan preferensi mereka. Untuk mengatasi tantangan ini, dibutuhkan sistem rekomendasi yang dapat secara otomatis menyarankan film berdasarkan minat pengguna.

Dalam proyek ini, dikembangkan sistem rekomendasi menggunakan **dua pendekatan**:
- **Content-Based Filtering**, yang memanfaatkan fitur seperti **judul** dan **genre** film untuk mencari kemiripan konten.
- **Collaborative Filtering berbasis deep learning**, yang mempelajari pola rating pengguna terhadap film, sehingga dapat merekomendasikan film berdasarkan preferensi pengguna lain yang serupa.

**Content-Based Filtering** bekerja dengan menganalisis karakteristik film yang pernah dipilih pengguna, kemudian merekomendasikan film lain yang memiliki fitur serupa. Sedangkan **Collaborative Filtering** memperhatikan pola interaksi antar pengguna, tanpa harus mengandalkan fitur film secara eksplisit.

### Mengapa Proyek Ini Penting?

Dengan meningkatnya volume konten digital, sistem rekomendasi menjadi alat penting untuk meningkatkan **pengalaman pengguna** dan **retensi pelanggan**. Platform yang mampu memberikan rekomendasi yang akurat cenderung memiliki tingkat engagement pengguna yang lebih tinggi.  
Selain itu, penerapan sistem rekomendasi juga terbukti berkontribusi signifikan dalam pertumbuhan bisnis di bidang hiburan digital.

Membangun sistem rekomendasi yang efektif, dengan kombinasi teknik content-based dan collaborative filtering, dapat meningkatkan kualitas personalisasi, memperluas eksplorasi pengguna terhadap berbagai jenis film, dan pada akhirnya meningkatkan loyalitas pengguna.

### Referensi Terkait

1. [A Survey on Recommender Systems: Research and Applications](https://scholar.google.com/scholar?q=A+survey+on+recommender+systems%3A+Research+and+applications) — Penelitian ini membahas berbagai pendekatan sistem rekomendasi, termasuk metode content-based dan collaborative filtering.
2. [Recommender Systems Handbook](https://scholar.google.com/scholar?q=Recommender+Systems+Handbook) — Buku komprehensif yang menjadi rujukan utama dalam pengembangan dan evaluasi sistem rekomendasi.
3. [A Survey of Recommender Systems Based on Deep Learning](https://arxiv.org/abs/2009.08544) — Studi ini mengulas pendekatan modern dalam sistem rekomendasi berbasis deep learning, termasuk model-model mutakhir untuk collaborative filtering.

### Pentingnya Proyek

Proyek ini dapat digunakan dalam berbagai skenario nyata seperti:
- Memberikan rekomendasi film yang lebih personal
- Membantu pengguna menemukan film baru yang relevan
- Meningkatkan engagement dan waktu tinggal pengguna pada platform

---
## Business Understanding

### Problem Statements

Dalam proyek ini, kita akan mengklarifikasi masalah utama yang ingin diselesaikan melalui pengembangan sistem rekomendasi film:

- **Pernyataan Masalah 1:** Banyak pengguna platform streaming mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka karena jumlah pilihan yang sangat besar.
- **Pernyataan Masalah 2:** Platform streaming memerlukan sistem rekomendasi yang efektif untuk menyarankan film berdasarkan genre atau jenis film yang pernah ditonton oleh pengguna.
- **Pernyataan Masalah 3:** Banyak film berkualitas tinggi yang terlewatkan oleh pengguna karena kurangnya sistem yang memperkenalkan film-film serupa dengan minat pengguna.

### Goals

Tujuan dari proyek ini adalah membangun sistem rekomendasi yang mampu membantu pengguna menemukan film yang relevan, memperkaya pengalaman menonton mereka, dan meningkatkan retensi pengguna platform streaming.

- **Jawaban Pernyataan Masalah 1:** Membuat sistem rekomendasi yang secara otomatis menyarankan film sesuai preferensi pengguna berdasarkan genre favorit mereka.
- **Jawaban Pernyataan Masalah 2:** Mengembangkan algoritma content-based filtering yang dapat memberikan rekomendasi berdasarkan kesamaan konten film (judul dan genre) yang pernah ditonton pengguna.
- **Jawaban Pernyataan Masalah 3:** Menyajikan rekomendasi film-film berkualitas tinggi yang relevan dengan preferensi pengguna, sehingga memperluas eksplorasi pengguna terhadap film baru.

### Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, digunakan dua pendekatan utama:

- **Pendekatan 1: Content-Based Filtering**
  - Sistem akan menganalisis fitur-fitur film, seperti **judul** dan **genre**, untuk mengidentifikasi film-film yang serupa dengan yang pernah ditonton pengguna.
  - Menggunakan teknik **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengekstrak representasi fitur teks, kemudian menghitung **cosine similarity** antar film untuk menentukan tingkat kemiripan.
  - Rekomendasi akan diberikan berdasarkan film-film yang memiliki skor kemiripan tertinggi dengan histori pengguna.

- **Pendekatan 2: Collaborative Filtering**
  - Selain content-based filtering, digunakan juga pendekatan **collaborative filtering berbasis deep learning** untuk meningkatkan akurasi rekomendasi.
  - Dengan memanfaatkan pola rating pengguna terhadap film, sistem dapat menemukan film-film yang disukai oleh pengguna lain dengan preferensi serupa.
  - Model dikembangkan menggunakan teknik **embedding** pengguna dan film, lalu dipelajari secara end-to-end menggunakan jaringan neural sederhana.

Dengan kombinasi kedua pendekatan ini, diharapkan sistem rekomendasi dapat memberikan pengalaman yang lebih personal dan akurat untuk setiap pengguna.

---
## Data Understanding

Dalam proyek ini, data yang digunakan berasal dari [MovieLens Dataset](https://grouplens.org/datasets/movielens/), sebuah dataset yang dikembangkan oleh GroupLens Research. MovieLens merupakan salah satu dataset benchmark yang banyak digunakan dalam penelitian sistem rekomendasi. Dataset ini berisi rating yang diberikan pengguna terhadap berbagai film, lengkap dengan metadata film seperti judul dan genre.

Dataset yang digunakan terdiri dari sekitar **100.000 rating** yang diberikan oleh lebih dari **600 pengguna** terhadap hampir **10.000 film**. Data ini sangat cocok untuk membangun dan menguji sistem rekomendasi baik berbasis konten maupun kolaboratif.

### Variabel-Variabel dalam Dataset

Variabel yang tersedia dalam dataset ini antara lain:

- **userId** : ID unik yang merepresentasikan masing-masing pengguna.
- **movieId** : ID unik yang merepresentasikan masing-masing film.
- **rating** : Nilai rating yang diberikan pengguna terhadap film (berkisar dari 0.5 hingga 5.0).
- **timestamp** : Waktu saat pengguna memberikan rating terhadap film (dalam format UNIX timestamp).
- **title** : Judul dari film yang dirating.
- **genre** : Genre film, dapat berupa kombinasi beberapa genre yang dipisahkan oleh tanda '|'.

### Exploratory Data Analysis (EDA)

Beberapa langkah eksplorasi data yang dilakukan dalam proyek ini adalah:

1. **Distribusi Rating Film**
   ![Distribusi Rating Film](https://github.com/GinantiRiski1/prediksi/blob/main/pic1.png)
   Visualisasi ini bertujuan untuk melihat sebaran nilai rating yang diberikan pengguna terhadap film.  
   - Hasil observasi menunjukkan bahwa sebagian besar film mendapatkan rating di kisaran 3.0 hingga 4.5.
   - Ini mengindikasikan bahwa pengguna cenderung memberikan penilaian positif terhadap film yang mereka tonton.

3. **Rata-rata Rating per Genre**
   ![Rata-rata Rating per Genre](https://github.com/GinantiRiski1/prediksi/blob/main/pic2.png)
   Untuk memahami preferensi berdasarkan genre, dilakukan analisis rata-rata rating per genre film.  
   - Genre dengan rata-rata rating tertinggi menunjukkan bahwa film dengan genre tersebut cenderung lebih disukai pengguna.
   - Dari visualisasi, dapat dilihat bahwa genre seperti **Documentary** dan **Film-Noir** mendapatkan rata-rata rating yang lebih tinggi dibanding genre lainnya.

### Insight dari EDA
- Sebagian besar pengguna memberikan rating yang relatif tinggi (di atas 3.0), yang berarti pengguna lebih banyak menilai film-film yang mereka sukai.
- Genre tertentu seperti **Documentary**, **Film-Noir**, dan **War** cenderung mendapatkan rating lebih tinggi dibanding genre lain, yang bisa menjadi informasi penting untuk meningkatkan personalisasi dalam sistem rekomendasi.

---
## Data Preparation

Pada bagian ini, kita akan menjelaskan tahapan-tahapan **data preparation** yang dilakukan untuk mempersiapkan data sebelum membangun sistem rekomendasi. Proses ini meliputi pemahaman terhadap data, pembersihan, transformasi data, dan persiapan variabel yang diperlukan untuk model content-based filtering dan collaborative filtering.

### Tahapan Data Preparation

1. **Data Understanding**
   - Data yang digunakan dalam proyek ini berasal dari empat file CSV yaitu `movies.csv`, `links.csv`, `ratings.csv`, dan `tags.csv`. Masing-masing file memiliki informasi yang berbeda, yaitu:
     - `movies.csv`: Berisi informasi mengenai film seperti `movieId`, `title`, dan `genres`.
     - `links.csv`: Menyediakan ID dari film yang terhubung dengan sumber eksternal (misalnya, IMDB).
     - `ratings.csv`: Berisi data rating yang diberikan oleh pengguna terhadap film.
     - `tags.csv`: Menyediakan tag atau label yang diberikan oleh pengguna pada film.
   - Sebelum melangkah lebih jauh, dilakukan **exploratory data analysis (EDA)** untuk memeriksa distribusi data dan variabel yang ada.

2. **Univariate EDA**
   - Analisis dilakukan untuk memeriksa distribusi rating film. Ini penting untuk memahami pola pemberian rating pengguna. Visualisasi distribusi rating dilakukan menggunakan histogram dan KDE untuk menunjukkan sebaran rating yang diberikan oleh pengguna.
   - Selain itu, rata-rata rating per genre juga dihitung dan divisualisasikan untuk mengidentifikasi genre dengan rating tertinggi secara keseluruhan.

3. **Data Processing**
   - **Pembersihan Data:** Beberapa baris yang memiliki data duplikat atau nilai yang tidak valid dibuang untuk memastikan data yang digunakan akurat.
   - **Transformasi Data:** Data yang berisi kolom `genres` diproses agar menjadi format yang dapat digunakan dalam model, dengan membagi genre menjadi daftar dan melakukan eksplosi data berdasarkan genre.

4. **Feature Engineering**
   - Fitur-fitur yang relevan untuk model rekomendasi dipilih dan diproses. Beberapa variabel yang penting adalah:
     - `movieId`: Identifikasi unik film.
     - `title`: Judul film.
     - `genres`: Genre film, yang diubah menjadi format daftar.
     - `rating`: Rating yang diberikan oleh pengguna terhadap film.

5. **Data Splitting**
   - Dataset dibagi menjadi data pelatihan dan data pengujian untuk memastikan model dapat diuji dengan baik pada data yang belum pernah dilihat sebelumnya.

6. **Persiapan untuk Content-Based Filtering**
   - Untuk membangun sistem rekomendasi berbasis konten, teknik **TF-IDF (Term Frequency-Inverse Document Frequency)** digunakan untuk mengukur pentingnya setiap kata dalam judul dan genre film. Ini membantu dalam membandingkan kesamaan konten antar film.
   - **Cosine similarity** digunakan untuk menghitung seberapa mirip film satu dengan yang lainnya berdasarkan TF-IDF score yang dihasilkan.

7. **Persiapan untuk Collaborative Filtering**
   - Dataset rating pengguna dipersiapkan untuk membangun model **Collaborative Filtering** menggunakan teknik **Matrix Factorization**. Data rating akan digunakan untuk menghitung kesamaan pengguna atau film berdasarkan interaksi yang ada.

### Mengapa Data Preparation Diperlukan?

Proses data preparation sangat penting untuk memastikan kualitas data yang digunakan dalam pembangunan model rekomendasi. Berikut adalah alasan mengapa setiap langkah penting:

- **Data Understanding** membantu untuk memahami konteks dan jenis data yang tersedia. Hal ini sangat penting sebelum melakukan analisis lebih lanjut.
- **Univariate EDA** memberikan wawasan tentang distribusi rating dan genre, yang akan membantu dalam memilih metode dan model yang tepat.
- **Data Processing** membersihkan data dari masalah-masalah seperti duplikasi atau nilai yang hilang, sehingga model yang dibangun tidak dipengaruhi oleh data yang tidak valid.
- **Feature Engineering** mengubah data mentah menjadi format yang dapat digunakan oleh algoritma machine learning dan memperkenalkan fitur-fitur baru yang relevan untuk prediksi.
- **Data Splitting** memastikan bahwa model diuji dengan data yang belum dilihat sebelumnya, yang sangat penting untuk menghindari overfitting dan mengukur performa model dengan baik.
- **Persiapan untuk Content-Based Filtering** memastikan bahwa fitur yang relevan untuk rekomendasi berdasarkan konten dapat digunakan dengan efisien, sehingga model dapat merekomendasikan film yang serupa.
- **Persiapan untuk Collaborative Filtering** memastikan bahwa interaksi pengguna dan film dapat digunakan untuk membangun model yang memahami preferensi pengguna secara lebih mendalam.

Dengan langkah-langkah tersebut, data siap digunakan untuk membangun sistem rekomendasi yang efektif dan akurat.

---
## Modeling

Pada tahapan ini, kami membangun dua jenis sistem rekomendasi untuk menyelesaikan permasalahan yang telah dijelaskan sebelumnya. Dua pendekatan yang digunakan adalah **Content-Based Filtering** dan **Collaborative Filtering**. Masing-masing pendekatan memiliki cara yang berbeda dalam memberikan rekomendasi film kepada pengguna berdasarkan data yang tersedia.

### 1. Content-Based Filtering

**Pendekatan:** Content-Based Filtering menganalisis konten film, seperti **judul** dan **genre**, untuk memberikan rekomendasi yang relevan bagi pengguna. Sistem ini bekerja dengan membandingkan kesamaan antara film yang telah ditonton oleh pengguna dengan film lainnya.

- **Proses:** 
    1. **Preprocessing:** Genre film diubah menjadi format daftar dan dihitung kesamaan antara film menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**.
    2. **Cosine Similarity:** Setelah mengubah genre menjadi representasi numerik, kami menggunakan **cosine similarity** untuk mengukur seberapa mirip dua film berdasarkan fitur genre dan judul mereka.
    3. **Rekomendasi:** Setelah menghitung kesamaan antar film, sistem akan memberikan rekomendasi berdasarkan film yang paling mirip dengan film yang sudah ditonton oleh pengguna.

- **Kelebihan:**
    - Tidak memerlukan data rating pengguna.
    - Mudah dipahami dan diimplementasikan, terutama untuk sistem dengan data film yang kaya fitur.
    - Cocok untuk pengguna baru karena rekomendasi diberikan berdasarkan konten yang ada, tanpa memerlukan interaksi pengguna sebelumnya.

- **Kekurangan:**
    - Tidak dapat memberikan rekomendasi yang sangat personal karena hanya mempertimbangkan konten, tanpa memperhitungkan preferensi individual pengguna.
    - Sistem ini dapat kekurangan keberagaman dalam rekomendasi karena hanya fokus pada kesamaan konten.

**Output - Top-N Content-Based Recommendations:** 
Setelah menghitung kesamaan, akan menampilkan 5 rekomendasi film berdasarkan pendekatan content-based filtering

### 2. Collaborative Filtering

**Pendekatan:** Collaborative Filtering mengandalkan data rating yang diberikan oleh pengguna untuk menemukan kesamaan antara pengguna dan memberikan rekomendasi berdasarkan preferensi pengguna lain yang mirip. Pendekatan ini lebih personal karena mempertimbangkan interaksi antar pengguna dan film.

- **Proses:**
    1. **Matrix Factorization:** Menggunakan teknik **Matrix Factorization** (misalnya, Singular Value Decomposition (SVD)) untuk mengidentifikasi pola preferensi pengguna berdasarkan rating film.
    2. **Prediksi Rating:** Setelah pemfaktoran matriks, sistem dapat memprediksi rating yang akan diberikan oleh pengguna pada film yang belum mereka tonton.
    3. **Rekomendasi:** Film dengan rating prediksi tertinggi yang belum ditonton oleh pengguna akan direkomendasikan.

- **Kelebihan:**
    - Memberikan rekomendasi yang lebih personal karena memperhitungkan preferensi pengguna lainnya.
    - Dapat memberikan variasi rekomendasi yang lebih besar karena mempertimbangkan interaksi pengguna dan film yang lebih banyak.

- **Kekurangan:**
    - Memerlukan data rating pengguna, sehingga tidak efektif untuk pengguna baru yang belum memberikan rating.
    - Dapat mengalami masalah **cold start** pada awalnya (yaitu, ketika data pengguna baru terlalu sedikit untuk memberikan rekomendasi yang akurat).
    - Lebih kompleks dalam implementasi dan membutuhkan lebih banyak data untuk menghasilkan rekomendasi yang baik.

**Output - Top-N Collaborative Recommendations:**
Dengan menggunakan Collaborative Filtering, akan menampilkan 10 rekomendasi film berdasarkan preferensi pengguna yang serupa

---
## Evaluation

Untuk mengevaluasi sistem rekomendasi yang telah dibangun, kami menggunakan beberapa metrik evaluasi yang sesuai dengan konteks masalah dan tujuan proyek. Metrik evaluasi yang digunakan adalah **Precision@10** untuk Content-Based Filtering dan **Root Mean Squared Error (RMSE)** untuk Collaborative Filtering.

### Metrik Evaluasi

1. **Precision@10** (Content-Based Filtering)
   Precision@10 mengukur seberapa banyak dari 10 rekomendasi teratas yang relevan dengan preferensi pengguna. Precision@10 dihitung dengan cara menghitung proporsi item yang relevan (misalnya, rating yang lebih tinggi) di antara 10 rekomendasi teratas. Formula untuk Precision@10 adalah:
Precision = TP / (TP + FP)

Dimana:
- **TP (True Positives)**: jumlah film yang direkomendasikan dan benar-benar relevan
- **FP (False Positives)**: jumlah film yang direkomendasikan tapi tidak relevan

### Hasil Evaluasi:
- **TP** = 5
- **FP** = 0
- **Precision** = 5 / (5 + 0) = **1.0 atau 100%**

   Precision@10 yang lebih tinggi menunjukkan bahwa model memberikan rekomendasi yang lebih relevan dan sesuai dengan preferensi pengguna. Dalam proyek ini, **Content-Based Filtering** mencapai **Precision@10 = 100%**, yang berarti semua rekomendasi teratas relevan dengan pengguna.

2. **Root Mean Squared Error (RMSE)** (Collaborative Filtering)
   RMSE mengukur akar kuadrat dari rata-rata kuadrat kesalahan antara nilai yang diprediksi dan nilai yang sebenarnya. Formula untuk RMSE adalah:

   \[
   RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{r}_i - r_i)^2}
   \]

   RMSE lebih sensitif terhadap outlier dan memberikan gambaran tentang kesalahan prediksi model. Semakin rendah nilai RMSE, semakin baik kualitas model dalam memberikan rekomendasi yang sesuai dengan preferensi pengguna.

   Dalam proyek ini, **Collaborative Filtering** memberikan hasil sebagai berikut:
   - **RMSE Training:** 0.1585
   - **RMSE Validation:** 0.1780

   RMSE Training menunjukkan bahwa rata-rata kesalahan prediksi model terhadap data training cukup kecil, sementara RMSE Validation menunjukkan bahwa model dapat generalisasi dengan baik pada data yang belum terlihat sebelumnya. Nilai RMSE yang mendekati antara training dan validation menunjukkan bahwa model tidak mengalami overfitting dan dapat memberikan prediksi yang akurat.

### Hasil Evaluasi

- **Content-Based Filtering:**
  - **Precision@10 = 100%** menunjukkan bahwa sistem rekomendasi berbasis konten memberikan rekomendasi yang sangat relevan kepada pengguna.

- **Collaborative Filtering:**
  - **RMSE Training (0.1585)** dan **RMSE Validation (0.1780)** menunjukkan bahwa model Collaborative Filtering dapat memprediksi rating dengan sangat akurat, dan kesalahan prediksi antara data training dan validation cukup kecil, mengindikasikan kemampuan generalisasi model yang baik.

### Kesimpulan

- **Content-Based Filtering**: Dengan Precision@10 yang mencapai 100%, sistem rekomendasi berbasis konten memberikan hasil yang sangat memuaskan dalam merekomendasikan film yang relevan dengan preferensi pengguna berdasarkan genre dan judul. Meskipun demikian, pendekatan ini hanya mempertimbangkan konten film dan tidak memperhitungkan interaksi antar pengguna.

- **Collaborative Filtering**: Dengan RMSE yang rendah untuk data training dan validation, sistem rekomendasi berbasis Collaborative Filtering dapat memberikan prediksi yang sangat akurat untuk pengguna berdasarkan interaksi dan preferensi pengguna lain. Pendekatan ini lebih personal dan efektif untuk pengguna dengan riwayat interaksi yang cukup banyak.

Kedua pendekatan menunjukkan hasil yang baik, dan dapat dikombinasikan untuk menghasilkan sistem rekomendasi yang lebih robust dan personalized bagi pengguna.
