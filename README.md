# Laporan Proyek Machine Learning (Sistem Rekomendasi Film Berbasis Konten Menggunakan Genre Film) - Ginanti Riski

## Project Overview

Sistem rekomendasi film merupakan salah satu penerapan machine learning yang paling luas digunakan saat ini, khususnya di industri hiburan seperti Netflix, Disney+, hingga platform streaming lokal. Sistem ini membantu pengguna menemukan film yang relevan berdasarkan preferensi mereka, sehingga dapat meningkatkan pengalaman pengguna dan loyalitas terhadap platform. Proyek ini berfokus pada pengembangan sistem rekomendasi film berbasis konten menggunakan genre film.

### Pentingnya Proyek

Proyek ini dapat digunakan dalam berbagai skenario nyata seperti:
- Memberikan rekomendasi film yang lebih personal
- Membantu pengguna menemukan film baru yang relevan
- Meningkatkan engagement dan waktu tinggal pengguna pada platform

### Referensi:
- [A Survey of Recommender Systems Based on Deep Learning](https://arxiv.org/abs/2009.08544)

## Business Understanding

### Problem Statements
1. Bagaimana membuat sistem rekomendasi film yang dapat memberikan saran film berdasarkan genre dari film yang sudah ditonton pengguna?
2. Bagaimana cara meningkatkan relevansi hasil rekomendasi?

### Goals
- Menghasilkan daftar 10 film rekomendasi berdasarkan input judul film.
- Menyusun sistem rekomendasi berbasis konten yang mudah diimplementasikan dan dievaluasi.

## Solution Approach

### Solution Statements:
1. Menggunakan **TF-IDF Vectorizer** pada fitur genre film untuk mengekstrak bobot representasi teks.
2. Menghitung **Cosine Similarity** antar film berdasarkan bobot genre.
3. Menyusun rekomendasi berdasarkan nilai kemiripan tertinggi terhadap film input (top-N recommendation).

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


