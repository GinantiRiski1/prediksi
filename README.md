# Laporan Proyek Machine Learning - Ginanti Riski

## Project Overview

Sistem rekomendasi film adalah aplikasi yang sangat populer digunakan oleh berbagai platform streaming untuk membantu pengguna menemukan film yang relevan berdasarkan preferensi mereka. Dalam proyek ini, kita akan membangun sistem rekomendasi menggunakan teknik content-based filtering. Sistem ini akan memanfaatkan fitur-fitur film seperti **judul** dan **genre** untuk memberikan rekomendasi yang paling relevan bagi pengguna.

Content-based filtering bekerja dengan cara menganalisis konten atau fitur dari item yang telah dilihat atau dipilih sebelumnya oleh pengguna dan mencari item lain yang memiliki kesamaan konten. Dalam hal ini, film yang memiliki genre atau judul yang mirip akan direkomendasikan kepada pengguna.

### Mengapa Proyek Ini Penting?

Seiring berkembangnya jumlah film dan serial di berbagai platform streaming seperti Netflix, Hulu, dan Amazon Prime, banyak pengguna merasa kesulitan untuk menemukan film yang sesuai dengan minat mereka. Sistem rekomendasi berbasis konten seperti ini memungkinkan pengguna untuk lebih mudah menemukan film baru yang mereka sukai tanpa harus mencari satu per satu. Oleh karena itu, membangun sistem rekomendasi yang efektif dan efisien menjadi sangat penting.

### Referensi Terkait
1. [A survey on recommender systems: Research and applications](https://scholar.google.com/scholar?q=A+survey+on+recommender+systems%3A+Research+and+applications) - Penelitian ini membahas berbagai jenis sistem rekomendasi, termasuk content-based filtering, yang sangat relevan dengan proyek ini.
2. [Recommender Systems Handbook](https://scholar.google.com/scholar?q=Recommender+Systems+Handbook) - Buku ini memberikan panduan menyeluruh tentang teori dan aplikasi sistem rekomendasi.
3. [A Survey of Recommender Systems Based on Deep Learning](https://arxiv.org/abs/2009.08544)

### Pentingnya Proyek

Proyek ini dapat digunakan dalam berbagai skenario nyata seperti:
- Memberikan rekomendasi film yang lebih personal
- Membantu pengguna menemukan film baru yang relevan
- Meningkatkan engagement dan waktu tinggal pengguna pada platform

--
## Business Understanding

### Problem Statements

Pada bagian ini, kita akan mengklarifikasi masalah yang ingin diselesaikan melalui proyek sistem rekomendasi film.

- **Pernyataan Masalah 1:** Banyak pengguna platform streaming merasa kesulitan dalam mencari film yang sesuai dengan preferensi mereka, karena banyaknya pilihan yang tersedia.
- **Pernyataan Masalah 2:** Platform streaming tidak memiliki sistem rekomendasi yang memadai untuk menyarankan film berdasarkan genre atau jenis film yang telah ditonton sebelumnya oleh pengguna.
- **Pernyataan Masalah 3:** Pengguna cenderung melewatkan film berkualitas tinggi karena kurangnya informasi terkait film yang serupa dengan preferensi mereka.

### Goals

Tujuan dari proyek ini adalah untuk membangun sistem rekomendasi berbasis konten yang dapat memberikan saran film yang relevan kepada pengguna berdasarkan genre dan judul film yang telah mereka tonton.

- **Jawaban Pernyataan Masalah 1:** Membuat sistem rekomendasi yang memungkinkan pengguna menemukan film yang relevan dengan lebih mudah, berdasarkan genre yang diminati.
- **Jawaban Pernyataan Masalah 2:** Membangun algoritma content-based filtering yang dapat memberikan rekomendasi berdasarkan kesamaan konten antara film yang telah ditonton dengan film lainnya.
- **Jawaban Pernyataan Masalah 3:** Menyajikan rekomendasi film yang dapat menarik minat pengguna dengan memperkenalkan mereka pada film-film berkualitas yang serupa dengan yang telah mereka tonton sebelumnya.

### Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, berikut adalah dua pendekatan solusi yang akan digunakan dalam proyek ini:

- **Pendekatan 1: Content-Based Filtering**
  - Sistem ini akan menganalisis fitur-fitur dari film, seperti judul dan genre, untuk merekomendasikan film serupa kepada pengguna. Setiap film akan diproses berdasarkan kesamaan genre dengan film yang telah dipilih atau ditonton sebelumnya.
  - Dalam pendekatan ini, kita akan menggunakan teknik **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengukur kesamaan antara film berdasarkan genre dan menggunakan **cosine similarity** untuk menghitung seberapa mirip film satu dengan yang lainnya.
  
- **Pendekatan 2: Collaborative Filtering**
  - Meskipun fokus utama adalah pada content-based filtering, pendekatan **Collaborative Filtering** bisa diterapkan sebagai solusi tambahan untuk meningkatkan kualitas rekomendasi, di mana sistem akan memanfaatkan data rating atau ulasan pengguna untuk menemukan film serupa yang disukai oleh pengguna lain dengan preferensi yang mirip.
  - Pendekatan ini dapat dikombinasikan dengan algoritma lain seperti **Matrix Factorization** untuk mendalami lebih jauh dalam membuat rekomendasi yang lebih personalized.

--


## Data Understanding

Dataset yang digunakan merupakan kumpulan film beserta genre-nya. Dataset terdiri dari 1000+ baris yang berisi informasi:
- **title**: judul film
- **genres**: genre film dalam bentuk teks yang telah diparsing (contoh: "action adventure fantasy")

### Contoh Data:
| title              | genres                                  |
|--------------------|-----------------------------------------|
| Avatar             | action adventure fantasy sciencefiction |
| The Dark Knight Rises | action crime drama thriller            |

Dataset ini tidak tersedia secara publik, namun merupakan hasil modifikasi dari dataset TMDB dan MovieLens yang dapat diunduh pada link berikut : https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system/input?select=tmdb_5000_movies.csv.

## Data Preparation

Tahapan data preparation yang dilakukan:
1. Cek Parsing kolom **genres** agar dalam bentuk teks string satu baris (misalnya dari list menjadi string dengan spasi).
2. **TF-IDF Vectorization** pada kolom genres untuk membuat representasi numerik dari fitur teks.
3. **Cosine Similarity Matrix** dihitung dari TF-IDF matrix antar film.

Langkah ini penting untuk membangun pondasi sistem rekomendasi berbasis kemiripan antar item.

## Modeling

Model sistem rekomendasi yang dibangun adalah model **content-based filtering** berdasarkan kemiripan genre. Implementasi dilakukan dengan tahapan:
1. Mengambil input judul film
2. Mengambil vektor genre dari film tersebut
3. Menghitung cosine similarity dengan semua film lain
4. Mengurutkan film berdasarkan nilai similarity tertinggi
5. Menampilkan top-N film (misalnya 20 film)

### Contoh Output:
**Input**: Superman

**Output**: Daftar 20 film yang memiliki genre mirip dengan Superman, lengkap dengan skor kemiripan.

Model ini mampu memberikan rekomendasi yang cukup relevan terutama untuk genre yang kuat (misalnya "action adventure" atau "sciencefiction").

## Evaluation

Metrik evaluasi yang digunakan adalah **Precision**, dihitung sebagai:
Precision = TP / (TP + FP)

Dimana:
- **TP (True Positives)**: jumlah film yang direkomendasikan dan benar-benar relevan
- **FP (False Positives)**: jumlah film yang direkomendasikan tapi tidak relevan

### Hasil Evaluasi:
- **TP** = 19
- **FP** = 0
- **Precision** = 19 / (19 + 0) = **1.0 atau 100%**

Metrik ini sesuai digunakan dalam sistem rekomendasi untuk mengevaluasi relevansi hasil rekomendasi. Evaluasi dilakukan secara manual terhadap output sistem dengan membandingkan genre hasil rekomendasi terhadap film input.

---

## Kesimpulan

Sistem rekomendasi berbasis konten ini berhasil memberikan rekomendasi yang sangat relevan berdasarkan genre film. Dengan pendekatan TF-IDF Vectorizer dan Cosine Similarity, sistem ini dapat mengidentifikasi film dengan genre yang serupa dengan akurasi yang tinggi. Evaluasi menggunakan Precision menunjukkan hasil yang sangat baik dengan skor 100%.


