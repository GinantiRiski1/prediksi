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

1. **Movie Recommendation System: A Comparison of Content-Based and Collaborative Filtering**  
   [Link ke jurnal](https://www.sciencedirect.com/science/article/pii/S1877050924031211)  
   **Penulis:** Hans Hendersen Kurniawan, William Susanto Lukman, Renaldy Fredyan, Muhammad Amien Ibrahim  
   **Hasil Penelitian:**  
   Penelitian ini membandingkan efektivitas content-based filtering dan collaborative filtering dalam membangun sistem rekomendasi film. Mereka menemukan bahwa **content-based filtering** lebih unggul untuk pengguna baru (cold-start problem), sementara **collaborative filtering** lebih efektif dalam menghasilkan rekomendasi personal yang lebih akurat untuk pengguna aktif. Kombinasi kedua metode tersebut memberikan performa yang lebih stabil dan fleksibel.

2. **Content Based Filtering and Collaborative Filtering: A Comparative Study**  
   [Link ke jurnal](https://www.researchgate.net/publication/378841543_Content_Based_Filtering_And_Collaborative_Filtering_A_Comparative_Study)  
   **Penulis:** Ms. Tejashri Sharad Phalle, Prof. Shivendu Bhushan  
   **Hasil Penelitian:**  
   Studi ini mengkaji kelebihan dan kekurangan masing-masing metode. **Content-based filtering** dinilai lebih cepat dan independen terhadap data pengguna lain, tetapi rentan terhadap "serendipity problem" (terlalu fokus pada kesamaan). Sementara **collaborative filtering** unggul dalam menawarkan rekomendasi yang lebih bervariasi namun membutuhkan data interaksi pengguna yang besar. Penelitian ini juga menyoroti pentingnya penggunaan model berbasis deep learning untuk meningkatkan akurasi collaborative filtering.

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

### Sumber Data
Data diambil dari dataset publik [MovieLens Dataset - GroupLens](https://grouplens.org/datasets/movielens/). Dataset ini merupakan salah satu benchmark utama dalam riset sistem rekomendasi.

---

### Informasi Dataset

Berikut adalah gambaran awal (raw data) dari masing-masing file yang digunakan:

| Nama File   | Jumlah Baris | Jumlah Kolom | Keterangan                     |
|-------------|--------------|--------------|--------------------------------|
| movies.csv  | 9.742        | 3            | movieId, title, genres         |
| links.csv   | 9.742        | 3            | movieId, imdbId, tmdbId        |
| tags.csv    | 40.108       | 4            | userId, movieId, tag, timestamp|
| ratings.csv | 100.836      | 4            | userId, movieId, rating, timestamp|

---

### Uraian Fitur

**movies.csv**
- `movieId`: ID unik untuk setiap film.
- `title`: Judul film.
- `genres`: Genre film (dapat lebih dari satu, dipisahkan dengan `|`).

**links.csv**
- `movieId`: ID film yang sama dengan `movies.csv`.
- `imdbId`: ID film di IMDb.
- `tmdbId`: ID film di TMDb.

**tags.csv**
- `userId`: ID pengguna.
- `movieId`: ID film.
- `tag`: Kata kunci atau label dari pengguna.
- `timestamp`: Waktu tag diberikan (format UNIX time).

**ratings.csv**
- `userId`: ID pengguna.
- `movieId`: ID film.
- `rating`: Penilaian pengguna terhadap film (skala 0.5 – 5.0).
- `timestamp`: Waktu pemberian rating.

---

### Kondisi Data (Raw)

#### Missing Values
- `movies.csv`, `links.csv`: Terdapat **2.221** nilai kosong pada kolom `title`, `genres`, dan ID eksternal.
- `tags.csv`: Terdapat **6.637** nilai kosong pada kolom `tag`.
- `ratings.csv`: Tidak ditemukan missing value.

#### Duplikat
- `ratings.csv`: Terdapat **8.946** baris duplikat.
- `tags.csv`: Terdapat duplikasi pada kombinasi `userId`, `movieId`, dan `tag`.

#### Outlier
- Kolom `rating`: Tidak ada outlier karena dibatasi sistem (0.5–5.0).
- Kolom `movieId` dan `tmdbId`: Nilai maksimum signifikan lebih tinggi dari Q3, perlu validasi lebih lanjut.
- Kolom `timestamp`: Distribusi wajar.

---

### Exploratory Data Analysis (EDA)

Beberapa langkah eksplorasi data yang dilakukan dalam proyek ini adalah:

1. **Distribusi Rating Film**
   
   ![Distirbusi Rating Film](https://github.com/user-attachments/assets/98de04b2-a342-4d72-a0db-7c8acebd62db)
   Visualisasi ini bertujuan untuk melihat sebaran nilai rating yang diberikan pengguna terhadap film.  
   - Hasil observasi menunjukkan bahwa sebagian besar film mendapatkan rating di kisaran 3.0 hingga 4.5.
   - Ini mengindikasikan bahwa pengguna cenderung memberikan penilaian positif terhadap film yang mereka tonton.

3. **Rata-rata Rating per Genre**
   ![Rata-rata Rating per Genre](https://github.com/user-attachments/assets/2f92cf00-1632-4492-83de-bcbc5ea6ad0c)

   Untuk memahami preferensi berdasarkan genre, dilakukan analisis rata-rata rating per genre film.  
   - Genre dengan rata-rata rating tertinggi menunjukkan bahwa film dengan genre tersebut cenderung lebih disukai pengguna.
   - Dari visualisasi, dapat dilihat bahwa genre seperti **Documentary** dan **Film-Noir** mendapatkan rata-rata rating yang lebih tinggi dibanding genre lainnya.

### Insight dari EDA
- Sebagian besar pengguna memberikan rating yang relatif tinggi (di atas 3.0), yang berarti pengguna lebih banyak menilai film-film yang mereka sukai.
- Genre tertentu seperti **Documentary**, **Film-Noir**, dan **War** cenderung mendapatkan rating lebih tinggi dibanding genre lain, yang bisa menjadi informasi penting untuk meningkatkan personalisasi dalam sistem rekomendasi.

---
## Data Preparation

Pada bagian ini dilakukan proses **data preparation** secara menyeluruh untuk mempersiapkan data MovieLens sebelum membangun sistem rekomendasi. Tahapan ini meliputi pembatasan jumlah data, penggabungan data, pembersihan dan seleksi fitur, transformasi struktur data, encoding, normalisasi, pemisahan data, serta ekstraksi fitur menggunakan TF-IDF. Semua proses ini bertujuan untuk memastikan kualitas data dan kesesuaian format input dengan pendekatan **Content-Based Filtering** maupun **Collaborative Filtering** menggunakan model **RecommenderNet**.

---

### 1. Pembatasan Jumlah Data

Karena ukuran data asli cukup besar, maka setiap file dibatasi hingga 10.000 baris data teratas agar proses eksplorasi dan pelatihan model lebih efisien.

**Mengapa penting?**  
Pembatasan ini bertujuan untuk menghindari beban komputasi yang tinggi dan mempercepat proses pengolahan data serta iterasi eksperimen selama tahap pengembangan model.

---

### 2. Penggabungan Data (Merge)

Proses penggabungan dilakukan agar seluruh informasi dari berbagai file saling terhubung. Langkah-langkah yang dilakukan meliputi:

- **Penggabungan seluruh `movieId`** dari `movies`, `links`, `ratings`, dan `tags`, lalu disortir dan dideduplikasi.
- **Penggabungan seluruh `userId`** dari `ratings` dan `tags`, lalu disortir dan dideduplikasi.
- **Penggabungan data film** (`movies`, `links`, dan `tags`) berdasarkan `movieId`.
- **Penggabungan dengan `ratings`** untuk menghubungkan informasi film dan pengguna dalam satu tabel.
- **Pembersihan kolom duplikat** hasil merge (`userId_y`, `timestamp_y`), lalu penggantian nama `userId_x` dan `timestamp_x` menjadi `userId` dan `timestamp`.
- **Simpan dataset gabungan (opsional)** ke file CSV.

**Mengapa penting?**  
Penggabungan ini menciptakan dataset yang terintegrasi dan kaya informasi, menjadi fondasi utama dalam pengembangan sistem rekomendasi berbasis konten maupun kolaboratif.

---

### 3. Seleksi Fitur dan Pembersihan Data

Setelah proses penggabungan, dilakukan pemilihan fitur penting dan penanganan data tidak bersih:

- Hanya kolom `userId`, `movieId`, `rating`, dan `timestamp` yang digunakan untuk model kolaboratif.
- Data film seperti `title` dan `genres` dipertahankan untuk model berbasis konten.
- Menghapus data duplikat berdasarkan `movieId`.

**Mengapa penting?**  
Pemilihan fitur yang tepat memastikan model hanya menggunakan informasi yang relevan, menghindari noise yang dapat menurunkan performa.

---

### 4. Transformasi dan Pembentukan Struktur Data

Langkah ini dilakukan untuk menyiapkan struktur data yang sesuai untuk proses rekomendasi.

#### 4.1 Salin dan Urutkan Berdasarkan `movieId`
Dataset `fix_movie` disalin ke variabel baru (`preparation`) dan diurutkan berdasarkan `movieId`.

**Mengapa penting?**  
Menjaga konsistensi urutan data sangat penting dalam proses pemodelan dan penyusunan vektor fitur.

#### 4.2 Hapus Duplikasi Berdasarkan `movieId`
Menghapus baris duplikat agar setiap film hanya direpresentasikan sekali.

**Mengapa penting?**  
Menghindari bias pada model akibat kemunculan ganda dari film yang sama.

#### 4.3 Konversi Menjadi List
Data bersih dikonversi ke dalam tiga list:
- `movie_id`
- `movie_title`
- `movie_genres`

**Mengapa penting?**  
List ini lebih efisien dan fleksibel untuk diolah dalam proses pembentukan vektor fitur dan tampilan rekomendasi.

#### 4.4 Pembentukan Dataframe Final
List di atas kemudian digabung kembali menjadi dataframe `movie_new` berisi tiga kolom: `id`, `title`, dan `genres`.

**Mengapa penting?**  
Dataframe ini menjadi dasar sistem rekomendasi berbasis konten yang memanfaatkan genre dan judul film.

---

### 5. Ekstraksi Fitur dengan TF-IDF

Langkah ini dilakukan untuk memproses teks genre menjadi representasi numerik yang bisa dihitung kemiripannya.

- Genre diformat ulang menjadi satu string per film.
- Diterapkan TF-IDF Vectorizer terhadap kolom genre.

**Mengapa penting?**  
TF-IDF memungkinkan kita menghitung pentingnya genre tertentu dalam koleksi film. Ini menjadi dasar dalam pengukuran kemiripan antar film untuk sistem rekomendasi berbasis konten, misalnya dengan cosine similarity.

---

### 6. Encoding dan Normalisasi untuk Collaborative Filtering

Langkah ini menyiapkan data agar bisa digunakan oleh model neural network (RecommenderNet).

#### 6.1 Encoding `userId` dan `movieId`
ID pengguna dan film diubah ke format numerik menggunakan mapping dictionary.

**Mengapa penting?**  
Model pembelajaran mesin memerlukan input numerik. Encoding ini juga membantu efisiensi memori dan performa model.

#### 6.2 Mapping Encoding ke Dataframe
Hasil encoding dimasukkan kembali ke dataframe sebagai kolom baru.

**Mengapa penting?**  
Pemetaan ke dataframe menjembatani data mentah dan input model pelatihan.

#### 6.3 Normalisasi Rating
Rating dinormalisasi ke rentang 0–1.

**Mengapa penting?**  
Normalisasi mempercepat konvergensi pelatihan dan menghindari dominasi nilai rating tinggi.

---

### 7. Pengacakan dan Penyimpanan Dataset

Dataset yang telah selesai diacak secara acak (shuffled) dan disimpan ke file CSV.

**Mengapa penting?**  
Shuffling mencegah urutan data mempengaruhi pelatihan model. Penyimpanan memungkinkan penggunaan ulang tanpa preprocessing ulang.

---

### 8. Pembentukan Input dan Output Model

Dataset dipisah menjadi:

- `x`: pasangan encoded `(userId, movieId)`
- `y`: rating yang telah dinormalisasi

Kemudian dilakukan split data menjadi 80% training dan 20% validation.

**Mengapa penting?**  
Format ini sesuai kebutuhan model RecommenderNet dan pemisahan data memastikan kemampuan generalisasi model bisa diuji.

---

## Modeling

Pada tahapan ini, saya membangun dua jenis sistem rekomendasi untuk menyelesaikan permasalahan yang telah dijelaskan sebelumnya. Dua pendekatan yang digunakan adalah **Content-Based Filtering** dan **Collaborative Filtering**. Masing-masing pendekatan memiliki cara yang berbeda dalam memberikan rekomendasi film kepada pengguna berdasarkan data yang tersedia.

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
Setelah menghitung kesamaan, sistem akan menampilkan 5 rekomendasi film berdasarkan pendekatan *content-based filtering*.  

Sebagai contoh, jika kita memilih film **Jumanji (1995)** dari dataset, maka hasil pengecekan data film adalah sebagai berikut:

| id | title           | genres                        |
|----|-----------------|-------------------------------|
| 1  | Jumanji (1995)  | Adventure &#124; Children &#124; Fantasy |


Setelah memilih film **Jumanji (1995)**, sistem akan mencari 5 film lain yang memiliki *genres* yang sama yaitu `Adventure | Children | Fantasy`, berdasarkan hasil fungsi *content-based filtering* yang telah dibuat.

Berikut adalah 5 rekomendasi film dengan genre yang sama:

| title                                             | genres                         |
|---------------------------------------------------|--------------------------------|
| Santa Claus: The Movie (1985)                     | Adventure &#124; Children &#124; Fantasy |
| NeverEnding Story, The (1984)                     | Adventure &#124; Children &#124; Fantasy |
| NeverEnding Story II: The Next Chapter, The (1990)| Adventure &#124; Children &#124; Fantasy |
| Escape to Witch Mountain (1975)                   | Adventure &#124; Children &#124; Fantasy |
| Return to Oz (1985)                               | Adventure &#124; Children &#124; Fantasy |

---

### 2. Collaborative Filtering

**Pendekatan:**  
Collaborative Filtering dalam proyek ini menggunakan pendekatan **Neural Collaborative Filtering**, bukan metode klasik seperti Matrix Factorization (SVD). Model dibangun menggunakan TensorFlow dan Keras dengan jaringan saraf tiruan yang mempelajari pola interaksi antara pengguna dan film.

#### Proses:

1. **Pembangunan Model:**  
   Model `RecommenderNet` dibuat dengan `tf.keras.Model`, berisi layer embedding untuk `user` dan `movie`, serta bias masing-masing.

2. **Representasi Embedding:**  
   Setiap `user` dan `movie` direpresentasikan sebagai vektor embedding berdimensi 50 yang dilatih bersamaan.

3. **Kalkulasi Prediksi:**  
   Embedding `user` dan `movie` dihitung menggunakan dot product dan ditambahkan dengan bias, kemudian diaktivasi dengan fungsi sigmoid untuk memprediksi rating (0–1).

4. **Training Model:**  
   Model dikompilasi dengan:
   - `loss`: `BinaryCrossentropy`
   - `optimizer`: `Adam(learning_rate=0.001)`
   - `metrics`: `RootMeanSquaredError`

   Proses training dilakukan dengan:
   - `epochs`: 100
   - `batch_size`: 8
   - `validation_data`: `(x_val, y_val)`

5. **Rekomendasi:**  
   Setelah model dilatih, prediksi rating dilakukan untuk semua film yang belum ditonton oleh pengguna, dan film dengan skor tertinggi direkomendasikan.

#### Kelebihan:
- Lebih fleksibel daripada Matrix Factorization karena bisa menangkap hubungan non-linear.
- Memberikan rekomendasi personal yang lebih akurat seiring bertambahnya data pengguna.

#### Kekurangan:
- Membutuhkan sumber daya komputasi dan waktu training yang lebih besar.
- Kurang efektif untuk pengguna baru yang belum memiliki data interaksi (cold start problem).

### Output - Top-N Collaborative Recommendations:

Dengan menggunakan **Collaborative Filtering**, berikut adalah rekomendasi 10 film berdasarkan preferensi pengguna yang serupa:

---
kita akan coba memasukkan sebuah dataset sample untuk menguji model yang sudah kita latih, dan hasil yang didapatkan adalah sebagai berikut :
**Pengguna dengan rating tertinggi untuk film:**
- **Dead Poets Society (1989)** : Drama
- **Spirited Away (Sen to Chihiro no kamikakushi) (2001)** : Adventure | Animation | Fantasy

**Top 10 Rekomendasi Film:**

| No | Judul Film                                                       | Genre                                       |
|----|------------------------------------------------------------------|---------------------------------------------|
| 1  | Heavy Metal (1981)                                               | Action | Adventure | Animation | Horror | Sci-Fi |
| 2  | Singin' in the Rain (1952)                                        | Comedy | Musical | Romance                |
| 3  | Dead Alive (Braindead) (1992)                                     | Comedy | Fantasy | Horror                |
| 4  | Rosencrantz and Guildenstern Are Dead (1990)                      | Comedy | Drama                            |
| 5  | Real Genius (1985)                                               | Comedy                                   |
| 6  | 400 Blows, The (Les quatre cents coups) (1959)                    | Crime | Drama                           |
| 7  | Monty Python's And Now for Something Completely Different (1971)  | Comedy                                   |
| 8  | Baraka (1992)                                                    | Documentary                              |
| 9  | Suspiria (1977)                                                  | Horror                                    |
| 10 | Brotherhood of the Wolf (Pacte des loups, Le) (2001)             | Action | Mystery | Thriller             |

---

## Evaluation

Untuk mengevaluasi sistem rekomendasi yang telah dibangun, kami menggunakan beberapa metrik evaluasi yang sesuai dengan konteks masalah dan tujuan proyek. Metrik evaluasi yang digunakan adalah **Precision@10** untuk Content-Based Filtering dan **Root Mean Squared Error (RMSE)** untuk Collaborative Filtering.

### Metrik Evaluasi

1. **Precision@10** (Content-Based Filtering)  
   Precision@10 mengukur seberapa banyak dari 10 rekomendasi teratas yang relevan dengan preferensi pengguna. Precision@10 dihitung dengan cara menghitung proporsi item yang relevan (misalnya, rating yang lebih tinggi) di antara 10 rekomendasi teratas. Formula untuk Precision@10 adalah:

   ![rumus precision](https://github.com/user-attachments/assets/5c44452d-c627-411b-9eaf-34d6e99104b4)

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
   
    ![Rumus RMSE](https://github.com/user-attachments/assets/f17fe4d4-6cfc-41ff-bf10-3216fc8fbddc)

   RMSE lebih sensitif terhadap outlier dan memberikan gambaran tentang kesalahan prediksi model. Semakin rendah nilai RMSE, semakin baik kualitas model dalam memberikan rekomendasi yang sesuai dengan preferensi pengguna.

   Dalam proyek ini, **Collaborative Filtering** memberikan hasil sebagai berikut:
   
   ![Hasil Evaluasi](https://github.com/user-attachments/assets/ac20f7b0-f2eb-4ddd-990e-6a3dce2b53cf)
   
   - **RMSE Training:** 0.1591
   - **RMSE Validation:** 0.1783

   RMSE Training menunjukkan bahwa rata-rata kesalahan prediksi model terhadap data training cukup kecil, sementara RMSE Validation menunjukkan bahwa model dapat generalisasi dengan baik pada data yang belum terlihat sebelumnya. Nilai RMSE yang mendekati antara training dan validation menunjukkan bahwa model tidak mengalami overfitting dan dapat memberikan prediksi yang akurat.

---

### Dampak terhadap Business Understanding

**Pernyataan Masalah 1:** Banyak pengguna platform streaming mengalami kesulitan dalam menemukan film yang sesuai dengan preferensi mereka karena jumlah pilihan yang sangat besar.

- **Jawaban:** Dengan menggunakan **Content-Based Filtering**, sistem dapat menyarankan film yang sesuai dengan genre atau jenis film yang telah ditonton pengguna sebelumnya, meningkatkan relevansi rekomendasi. **Precision@10 = 100%** menunjukkan bahwa rekomendasi yang diberikan sangat relevan dengan preferensi pengguna, menjawab masalah ini secara efektif.

**Pernyataan Masalah 2:** Platform streaming memerlukan sistem rekomendasi yang efektif untuk menyarankan film berdasarkan genre atau jenis film yang pernah ditonton oleh pengguna.

- **Jawaban:** **Content-Based Filtering** menggunakan analisis genre dan kesamaan konten film untuk memberikan rekomendasi, yang memberikan hasil yang sangat relevan. Dengan **Precision@10 = 100%**, sistem berhasil memberikan rekomendasi berdasarkan genre yang sesuai dengan preferensi pengguna.

**Pernyataan Masalah 3:** Banyak film berkualitas tinggi yang terlewatkan oleh pengguna karena kurangnya sistem yang memperkenalkan film-film serupa dengan minat pengguna.

- **Jawaban:** **Collaborative Filtering** menggunakan pola rating pengguna lain untuk menyarankan film yang lebih banyak dan bervariasi. Dengan nilai **RMSE yang rendah (0.1591 pada training dan 0.1783 pada validation)**, model ini dapat menggeneralisasi dengan baik dan memberikan rekomendasi film berkualitas tinggi yang relevan dengan preferensi pengguna.

---

### Goals

Tujuan dari proyek ini adalah membangun sistem rekomendasi yang mampu membantu pengguna menemukan film yang relevan, memperkaya pengalaman menonton mereka, dan meningkatkan retensi pengguna platform streaming.

- **Jawaban:** Dengan **Content-Based Filtering** yang mencapai **Precision@10 = 100%** dan **Collaborative Filtering** yang menunjukkan nilai **RMSE rendah**, sistem berhasil mencapai tujuan untuk meningkatkan relevansi rekomendasi film dan memberikan pengalaman menonton yang lebih personal. Model mampu memperkenalkan film berkualitas tinggi yang relevan dengan preferensi pengguna, sekaligus mengurangi kesulitan dalam menemukan film yang sesuai.

---

### Hasil Evaluasi

- **Content-Based Filtering:**
  - **Precision@10 = 100%** menunjukkan bahwa sistem rekomendasi berbasis konten memberikan rekomendasi yang sangat relevan kepada pengguna, menjawab pernyataan masalah pertama dan kedua dengan efektif.
  
- **Collaborative Filtering:**
  - **RMSE Training (0.1591)** dan **RMSE Validation (0.1783)** menunjukkan bahwa model Collaborative Filtering dapat memprediksi rating dengan sangat akurat, mengurangi kesulitan dalam menemukan film berkualitas tinggi yang terlewatkan, dan membantu memperkenalkan film baru yang relevan, sesuai dengan pernyataan masalah ketiga.

Secara keseluruhan, sistem rekomendasi ini berhasil memenuhi tujuan proyek dan memberikan dampak positif terhadap pengalaman pengguna dalam menemukan film yang relevan dengan preferensi mereka.


### Kesimpulan

- **Content-Based Filtering**: Dengan Precision@10 yang mencapai 100%, sistem rekomendasi berbasis konten memberikan hasil yang sangat memuaskan dalam merekomendasikan film yang relevan dengan preferensi pengguna berdasarkan genre dan judul. Meskipun demikian, pendekatan ini hanya mempertimbangkan konten film dan tidak memperhitungkan interaksi antar pengguna.

- **Collaborative Filtering**: Dengan RMSE yang rendah untuk data training dan validation, sistem rekomendasi berbasis Collaborative Filtering dapat memberikan prediksi yang sangat akurat untuk pengguna berdasarkan interaksi dan preferensi pengguna lain. Pendekatan ini lebih personal dan efektif untuk pengguna dengan riwayat interaksi yang cukup banyak.

Kedua pendekatan menunjukkan hasil yang baik, dan dapat dikombinasikan untuk menghasilkan sistem rekomendasi yang lebih robust dan personalized bagi pengguna.
