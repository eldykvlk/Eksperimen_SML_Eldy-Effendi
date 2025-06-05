# Proses Analisis dan Prediksi Penjualan Walmart tahap Preprocessing

Berikut adalah alur lengkap proses analisis dan prediksi penjualan mingguan Walmart menggunakan dataset dari Kaggle: Walmart Sales (https://www.kaggle.com/datasets/mikhail1681/walmart-sales). Proses ini mencakup pengenalan dataset, impor pustaka, pemuatan data, eksplorasi data, preprocessing, pembagian data, dan ekspor hasil untuk keperluan pemodelan machine learning.

## 1. Perkenalan Dataset
Dataset ini berisi informasi historis penjualan mingguan dari berbagai toko Walmart, dilengkapi dengan variabel eksternal yang dapat memengaruhi penjualan. Kolom-kolom dalam dataset adalah sebagai berikut:
- Store: Nomor identifikasi unik untuk setiap toko.
- Date: Tanggal awal minggu penjualan.
- Weekly_Sales: Total penjualan mingguan (dalam USD).
- Holiday_Flag: Indikator liburan (1 untuk minggu dengan hari libur, 0 untuk non-libur).
- Temperature: Suhu rata-rata di wilayah toko (dalam Fahrenheit).
- Fuel_Price: Harga bahan bakar di wilayah toko (USD per galon).
- CPI: Indeks Harga Konsumen (indikator inflasi).
- Unemployment: Tingkat pengangguran di wilayah toko (%).

## 2. Import Library
Pustaka Python yang digunakan meliputi:
- pandas dan numpy: Untuk manipulasi data dan perhitungan numerik.
- sklearn.model_selection: Untuk train_test_split (membagi data) dan GridSearchCV (optimasi hyperparameter).
- sklearn.tree: DecisionTreeRegressor untuk model prediksi nilai kontinu.
- sklearn.metrics: mean_absolute_error, mean_squared_error, dan r2_score untuk evaluasi model.
- matplotlib.pyplot: Untuk visualisasi data dan hasil prediksi.
Fungsi utama meliputi preprocessing, pelatihan model, optimasi hyperparameter, evaluasi performa, dan visualisasi hasil.

## 3. Memuat Dataset
Dataset dimuat dari file CSV (Walmart_Sales_raw.csv) menggunakan pandas. Langkah ini mencakup:
- Membaca file ke dalam DataFrame.
- Menampilkan 5 baris pertama untuk memeriksa struktur data.
- Mengecek dimensi dataset (jumlah baris dan kolom).
- Menangani error seperti file tidak ditemukan atau format salah untuk memastikan data dimuat dengan benar.

## 4. Exploratory Data Analysis (EDA)
EDA dilakukan untuk memahami karakteristik dataset, mendeteksi anomali, dan merumuskan langkah preprocessing. Langkah-langkah utama:
- Pengecekan tipe data dan konversi kolom Date ke format datetime.
- Identifikasi missing values (tidak ada nilai kosong pada dataset ini).
- Analisis distribusi variabel numerik seperti Weekly_Sales (right-skewed, puncak $500k-$1.5M), Temperature (range 6.24°F–100.14°F), Fuel_Price ($2.50–$4.50), CPI, dan Unemployment (distribusi bimodal).
- Penanganan outlier menggunakan metode IQR: menghitung batas bawah (Q1 - 1.5*IQR) dan atas (Q3 + 1.5*IQR), lalu meng-clip nilai ekstrem.
- Visualisasi distribusi menggunakan histogram untuk setiap variabel numerik dan bar plot untuk Holiday_Flag (85% non-libur, 15% libur).
- Analisis korelasi menunjukkan Weekly_Sales memiliki korelasi negatif lemah dengan Store (-0.337), Unemployment (-0.109), CPI (-0.073), dan Temperature (-0.061).
Hasil EDA menunjukkan data cukup bersih, dengan outlier yang telah ditangani dan distribusi yang lebih terpusat setelah winsorization.

## 5. Data Preprocessing
Preprocessing dilakukan untuk memastikan data siap digunakan dalam model machine learning. Langkah-langkahnya meliputi:
- Konversi kolom Date ke format datetime untuk ekstraksi fitur temporal.
- Pengecekan missing values (tidak ditemukan).
- Feature engineering:
  - Ekstraksi Month (1-12) dan Year dari kolom Date untuk menangkap pola musiman dan tren tahunan.
  - Pembuatan lag features: Weekly_Sales_Lag1 (penjualan 1 minggu sebelumnya), Weekly_Sales_Lag2 (2 minggu sebelumnya), dan Weekly_Sales_Lag4 (4 minggu sebelumnya) untuk menangkap autokorelasi. Nilai NaN pada lag features ditangani dengan backward fill per toko.
  - Pembuatan interaction term: Temperature_Fuel_Interaction (perkalian Temperature dan Fuel_Price) untuk menangkap efek nonlinear.
- Penghapusan kolom Date setelah ekstraksi fitur.
Hasilnya adalah DataFrame baru (df_features) dengan kolom tambahan: Month, Year, Weekly_Sales_Lag1/2/4, dan Temperature_Fuel_Interaction.

## 6. Data Splitting
Data dibagi menjadi set pelatihan (training) dan pengujian (testing) untuk keperluan pelatihan dan validasi model:
- Fitur (X): Semua kolom kecuali Weekly_Sales.
- Target (y): Kolom Weekly_Sales.
- Pembagian: 80% untuk training dan 20% untuk testing menggunakan train_test_split dengan parameter:
  - test_size=0.2.
  - random_state=42 untuk hasil yang konsisten.
  - stratify=df_features['Store'] untuk menjaga distribusi toko seimbang di kedua set.
Hasilnya adalah X_train, X_test, y_train, dan y_test yang siap digunakan untuk pelatihan model.

## 7. Ekspor Data Hasil Preprocessing
Data yang telah di-preprocess diekspor ke file CSV baru (Walmart_Sales_preprocessing.csv) untuk analisis atau pemodelan lanjutan. Proses ini mencakup:
- Menyimpan DataFrame (df_features) tanpa indeks.
- Penanganan error seperti masalah izin akses atau direktori salah untuk memastikan ekspor berhasil.

## Ringkasan Proses
1. Dataset dimuat dan struktur datanya dipahami.
2. EDA dilakukan untuk menganalisis distribusi, korelasi, dan menangani outlier.
3. Preprocessing mencakup konversi tipe data, feature engineering, dan penanganan missing values.
4. Data dibagi menjadi training (80%) dan testing (20%) dengan stratifikasi berdasarkan toko.
5. Data yang telah di-preprocess diekspor ke file CSV untuk digunakan dalam pemodelan.
Dataset kini siap untuk pelatihan model machine learning, seperti DecisionTreeRegressor, dengan potensi optimasi hyperparameter menggunakan GridSearchCV.
