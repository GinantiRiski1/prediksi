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

Dalam proyek ini, data yang digunakan berasal dari [MovieLens Dataset](https://grouplens.org/datasets/movielens/), sebuah dataset yang dikembangkan oleh GroupLens Research. MovieLens merupakan salah satu dataset benchmark yang banyak digunakan dalam penelitian sistem rekomendasi. Dataset ini berisi rating yang diberikan pengguna terhadap berbagai film, lengkap dengan metadata film seperti judul dan genre. Data yang digunakan dalam proyek ini berasal dari empat file CSV yaitu `movies.csv`, `links.csv`, `ratings.csv`, dan `tags.csv`. 

Masing-masing file memiliki informasi yang berbeda, yaitu:
     - `movies.csv`: Berisi informasi mengenai film seperti `movieId`, `title`, dan `genres`.
     - `links.csv`: Menyediakan ID dari film yang terhubung dengan sumber eksternal (misalnya, IMDB).
     - `ratings.csv`: Berisi data rating yang diberikan oleh pengguna terhadap film.
     - `tags.csv`: Menyediakan tag atau label yang diberikan oleh pengguna pada film.


## Informasi Umum Dataset
- **Jumlah data**:
  - **46573 baris (entri rating)**
  - **9 kolom (fitur)**
- **Ukuran memori**: ~3.2 MB


## Kondisi Data

### Missing Values
| Kolom     | Nilai Non-Null | Nilai Kosong |
|-----------|----------------|---------------|
| title     | 44352          | 2221          |
| genres    | 44352          | 2221          |
| imdbId    | 44352          | 2221          |
| tmdbId    | 44351          | 2222          |
| tag       | 39936          | 6637          |

> Sebagian besar nilai kosong berada di kolom metadata film (`title`, `genres`, `imdbId`, `tmdbId`) dan `tag`, namun kita tidak akan menggunakan variabel `imdbid`, `tag` dan `tmdbid` nantinya.

### Duplikasi
- Terdapat **8946 baris data duplikat** pada keseluruhan data dalam dataset.

## Penjelasan Fitur

| Fitur        | Deskripsi |
|--------------|-----------|
| **userId**   | ID unik pengguna. |
| **movieId**  | ID unik film. |
| **rating**   | Rating dari pengguna terhadap film (0.5 - 5.0). |
| **timestamp**| Waktu rating diberikan (UNIX time). |
| **title**    | Judul film. |
| **genres**   | Genre film. |
| **imdbId**   | ID film versi IMDb. |
| **tmdbId**   | ID film versi TMDb. |
| **tag**      | Tag/kata kunci dari pengguna (opsional). |


## Statistik Deskriptif

| Fitur     | Min    | Q1     | Median | Q3     | Max      |
|-----------|--------|--------|--------|--------|----------|
| userId    | 1      | 21     | 33     | 49     | 66       |
| movieId   | 1      | 480    | 1274   | 4020   | 291485   |
| rating    | 0.5    | 3.0    | 4.0    | 5.0    | 5.0      |
| timestamp | 8.3e+08| 9.9e+08| 1.1e+09| 1.4e+09| 1.7e+09  |
| imdbId    | 9018   | 102926 | 111161 | 137523 | 436727   |
| tmdbId    | 5      | 238    | 510    | 854    | 503475   |


## Analisis Outlier

- **Rating**: Tidak ditemukan outlier karena seluruh nilai berada dalam rentang sistem rating resmi.
- **movieId** dan **tmdbId** memiliki nilai maksimum jauh di atas Q3. Perlu investigasi apakah ID tersebut sah atau noise.
- **timestamp** memiliki rentang waktu yang besar tetapi masih masuk akal.
- **userId** normal dan tidak menunjukkan penyimpangan.

> Catatan: Perlu dilakukan pembersihan data lebih lanjut terutama pada baris duplikat dan kolom dengan banyak nilai kosong, namun hanya berdasarkan fitur yang akan kita gunakan saja, yaitu (`userId`, `movieId`,`rating`,`title`,`genres` dan `timestamp`.

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

Pada bagian ini dilakukan proses **data preparation** secara menyeluruh untuk mempersiapkan data MovieLens sebelum membangun sistem rekomendasi. Tahapan ini meliputi: eksplorasi awal, penggabungan data, pembersihan data, transformasi fitur, encoding, normalisasi, serta pemisahan data. Proses ini dilakukan untuk memastikan kualitas data yang digunakan serta menyiapkan format data yang sesuai dengan pendekatan **Content-Based Filtering** dan **Collaborative Filtering (menggunakan RecommenderNet)**.

## Tahapan Data Preparation :

### 1. Pembatasan Jumlah Data dan Eksplorasi Awal

Karena ukuran data asli cukup besar dan untuk menjaga efisiensi komputasi selama proses eksplorasi dan pelatihan model, maka setiap file dibatasi hingga 10.000 baris data teratas. dan eksplorasi awal kita lakukan dengan melihat struktur data disetiap dataset.

**Mengapa penting?**  
Pembatasan jumlah data dilakukan untuk menghindari beban komputasi yang terlalu tinggi dan mempercepat proses analisis awal serta pengembangan model. Hal ini juga memudahkan pemahaman terhadap struktur data dan mempercepat iterasi saat eksplorasi serta preprocessing. eksplorasi awal dilakukan untuk mengetahui informasi dataset dan fitur yang ada didalamnya.

---
### 2. Penggabungan Data (Merge)

Beberapa file yang berbeda digunakan untuk membentuk satu dataset utama:

- Penggabungan seluruh movieId
Data movieId dari movies, links, ratings, dan tags digabungkan menjadi satu, diurutkan, dan duplikasi data dihapus agar didapatkan seluruh ID film yang unik.

- Penggabungan seluruh userId
Begitu juga userId dari ratings dan tags digabung, duplikasi data dihapus agar didapatkan seluruh ID film yang unik, lalu data diurutkan secara ascending.

- Penggabungan Data Film
Dataset movies digabungkan dengan links dan tags berdasarkan movieId untuk menambahkan metadata film dan tag pengguna.

- Penggabungan dengan Ratings
Selanjutnya, ratings digabung dengan hasil penggabungan data film sehingga terbentuk satu dataset yang lengkap mencakup informasi user, movie, rating, tag, genre, dan identifier eksternal.

- Pembersihan Kolom Duplikat
Setelah merge, kolom duplikat seperti userId_y dan timestamp_y dihapus, lalu kolom userId_x dan timestamp_x diubah namanya menjadi userId dan timestamp.

- Simpan Dataset Gabungan (Opsional)
Hasil akhir penggabungan data disimpan dalam file CSV agar bisa digunakan ulang tanpa perlu proses merge ulang.

**Mengapa penting?**
Penggabungan ini menciptakan dataset terintegrasi yang berisi informasi lengkap untuk setiap kombinasi userId dan movieId. Hal ini menjadi fondasi utama dalam pengembangan sistem rekomendasi berbasis konten maupun kolaboratif.

---
### 3: Pembersihan dan Seleksi Fitur

Langkah ini merupakan tahap krusial dalam proses pengolahan data karena bertujuan untuk memastikan bahwa data yang digunakan bersih, relevan, dan siap digunakan untuk proses selanjutnya seperti analisis atau pemodelan.

#### 3.1 Cek Statistik Deskriptif Data

Langkah pertama yaitu melihat statistik deskriptif untuk mengetahui adanya nilai ekstrem atau outliers yang dapat mempengaruhi hasil analisis dan pemodelan.

**Mengapa penting?**
Outliers dapat menyebabkan bias pada hasil analisis dan menurunkan performa model. Dengan describe(), kita dapat melihat distribusi data dan mengidentifikasi nilai-nilai yang mencurigakan.

#### 3.2 Cek Missing Values
Langkah kedua adalah memeriksa apakah terdapat nilai kosong (missing values) dalam dataset.

**Mengapa penting?**
Nilai kosong bisa menyebabkan error saat pelatihan model atau menghasilkan prediksi yang tidak akurat. Data yang tidak lengkap perlu ditangani agar tidak mempengaruhi integritas analisis.

#### 3.3 Cek Duplikasi Data
Langkah selanjutnya adalah memeriksa data yang duplikat dan menghapusnya jika ditemukan.

**Mengapa penting?**
Duplikasi data bisa menyebabkan model menjadi bias, karena beberapa informasi akan dianggap lebih penting hanya karena muncul lebih sering dari seharusnya.

#### 3.4 Seleksi Fitur yang Relevan
Hanya kolom-kolom penting yang digunakan untuk proses selanjutnya, yaitu userId, movieId, rating, dan timestamp. Selanjutnya, data tersebut digabungkan dengan data film (judul dan genre).

**Mengapa penting?**
Memilih fitur yang tepat membuat proses analisis lebih efisien dan mengurangi noise pada model. Penggabungan data dengan metadata film juga memperkaya informasi untuk analisis konten dan rekomendasi.

#### 3.5 Cek Ulang Missing Values Setelah Gabung dan Hapus
Setelah proses penggabungan data, dilakukan kembali pengecekan nilai kosong dan dihapus jika ada.

**Mengapa penting?**
Penggabungan data seringkali menghasilkan nilai kosong (jika data tidak cocok sepenuhnya). Menghapus data kosong memastikan integritas data tetap terjaga.

#### 3.6 Menyamakan Jenis Genre dan Mengurutkan Berdasarkan `movieId`
Langkah terakhir dalam tahap ini adalah mengurutkan data berdasarkan `movieId` agar lebih rapi dan konsisten.

**Mengapa penting?**
Mengurutkan data mempermudah analisis selanjutnya dan memastikan data konsisten dalam urutan, terutama untuk pemrosesan batch atau pembuatan indeks.

---
### 4: Transformasi dan Pembentukan Struktur Data
Pada tahap ini dilakukan transformasi data dari dataframe hasil penggabungan (`fix_movie`) menjadi struktur data yang lebih terorganisir dan siap digunakan untuk proses pemodelan sistem rekomendasi, terutama untuk pendekatan berbasis konten.

#### 4.1 Salin dan Urutkan Data berdasarkan movieId
Data dari `fix_movie` disalin ke dalam variabel baru bernama preparation, lalu diurutkan berdasarkan kolom movieId untuk memastikan keteraturan dan konsistensi dalam pemrosesan data.

**Mengapa penting?**
Pengurutan data membantu memastikan konsistensi dalam pemrosesan berikutnya, terutama saat mengonversi data ke dalam bentuk list atau dictionary, yang sangat bergantung pada urutan indeks.

#### 4.2 Menghapus Duplikasi Data berdasarkan movieId
Langkah selanjutnya adalah membuang data duplikat berdasarkan kolom `movieId` untuk menjaga hanya satu representasi per film.

**Mengapa penting?**
Duplikasi data bisa menyebabkan bias saat membangun model rekomendasi, karena film yang sama bisa dianggap lebih penting dari yang lain jika muncul berulang.

#### 4.3 Konversi Data menjadi List
Data yang sudah bersih dikonversi ke dalam tiga list terpisah:
- `movie_id`: berisi daftar ID film.
- `movie_title`: berisi daftar judul film.
- `movie_genres`: berisi daftar genre film.

**Mengapa penting?**
List ini akan digunakan untuk membuat struktur data yang lebih efisien dan mudah dimanipulasi dalam proses rekomendasi, baik untuk keperluan tampilan, pemrosesan konten, maupun pembentukan vektor fitur.

#### 4.4 Pembentukan Dataframe Final
List yang telah dibuat kemudian digabung kembali menjadi dataframe baru bernama movie_new yang terdiri dari tiga kolom utama: `id`, `title`, dan `genres`.

**Mengapa penting?**
Dataframe ini menjadi basis dari sistem rekomendasi berbasis konten, di mana genre dan judul film akan digunakan untuk menghitung kemiripan antar film (misalnya dengan TF-IDF atau cosine similarity).

---
### 5: Encoding dan Persiapan Data untuk Model Collaborative Filtering
Langkah ini mempersiapkan data untuk dimasukkan ke dalam model Collaborative Filtering berbasis neural network, seperti RecommenderNet. Proses ini mencakup encoding ID pengguna dan film, normalisasi rating, hingga pembagian data menjadi data latih dan validasi.

#### 5.1 Seleksi Kolom Rating
Dataset difokuskan pada kolom userId, movieId, rating, dan timestamp, karena kolom ini diperlukan untuk model rekomendasi berbasis interaksi pengguna dan film.

**Mengapa penting?**
- Fokus pada kolom userId, movieId, rating, dan timestamp memungkinkan kita untuk menyaring hanya data yang relevan.
- Kolom-kolom ini merupakan inti interaksi pengguna dalam sistem rekomendasi berbasis Collaborative Filtering, di mana sistem belajar dari kebiasaan pengguna memberikan rating terhadap film tertentu.

#### 5.2 Encoding userId dan movieId
Agar bisa digunakan dalam model machine learning, userId dan movieId perlu diubah menjadi format numerik yang efisien melalui proses encoding.

**Mengapa penting?**
- Algoritma machine learning tidak bisa bekerja langsung dengan data bertipe string atau ID unik.
- Dengan mengubah ID menjadi angka melalui encoding, kita dapat memanfaatkan model pembelajaran seperti neural network yang hanya menerima input numerik.
- Encoding juga membantu dalam membangun representasi yang efisien dan memori-friendly.
  
#### 5.3 Mapping Encoding ke DataFrame
Setelah membuat kamus encoding, kita mapping ke dataframe agar bisa digunakan untuk pelatihan model.

**Mengapa penting?**
- Setelah encoding dibuat, perlu dilakukan pemetaan ke dalam dataframe agar bisa digunakan sebagai input model.
- Proses ini mengubah bentuk data dari bentuk original ke bentuk numerik yang siap dilatih, menjembatani data mentah ke input model.

#### 5.4 Statistik Dataset dan Normalisasi Rating
Kita hitung jumlah pengguna, jumlah film, dan rentang nilai rating. Rating juga dinormalisasi ke dalam rentang 0-1 agar cocok untuk output neural network.

**Mengapa penting?**
- Mengetahui jumlah user, jumlah film, dan rentang rating sangat penting untuk:
  - Menentukan dimensi input dan output model.
  - Menyusun arsitektur embedding dan normalisasi output.
- Normalisasi rating ke rentang 0–1 membantu model neural network dalam mempercepat konvergensi dan mencegah bias terhadap nilai rating besar.

#### 5.5 Pengacakan Data dan Simpan ke CSV
Dataset diacak secara acak dan disimpan sebagai file CSV untuk digunakan ulang.

**Mengapa penting?**
- Pengacakan dataset (shuffling) memastikan bahwa data yang digunakan untuk pelatihan dan validasi tidak bias terhadap urutan aslinya.
- Menyimpan dataset ke CSV membuat proses pelatihan dapat diulang atau digunakan kembali tanpa harus melakukan preprocessing dari awal hingga efisien untuk eksperimen berulang.

#### 5.6 Persiapan Input dan Output Model
Untuk model rekomendasi, kita buat:
- x: kombinasi pasangan user dan movie
- y: rating yang sudah dinormalisasi

Kemudian dataset dibagi menjadi 80% data latih dan 20% data validasi.

**Mengapa penting?**
- Model rekomendasi membutuhkan input berupa pasangan (user, movie) agar bisa mempelajari hubungan antara pengguna dan film.
- Rating sebagai output model harus dalam bentuk numerik dan terstandarisasi.
- Membagi dataset menjadi train dan validation (misal 80:20) sangat penting untuk:
  - Melatih model secara efisien
  - Mengukur performa model pada data yang belum pernah dilihat, yang membantu mencegah overfitting.

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
