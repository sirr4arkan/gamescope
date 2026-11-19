# PRODUCT REQUIREMENTS DOCUMENT (PRD)
**Judul Proyek:** Sistem Cerdas Prediksi Penjualan Global dan Analisis Pasar Industri Video Game Berbasis Machine Learning  
**Mata Kuliah:** Sistem Cerdas (TI24317) — Pertemuan 14 & Proyek Akhir UAS  
**Program Studi:** D3 Teknologi Informasi, Politeknik Negeri Madiun  
**Arsitektur Teknologi:** Python, Scikit-Learn, Django (Backend), SQLite lokal (database tahap awal), Streamlit (Frontend)  

---

## 1. Latar Belakang & Tujuan Proyek
### 1.1 Latar Belakang
Dalam pengembangan produk perangkat lunak cerdas, model *Machine Learning* yang hanya berjalan di dalam lingkungan *Jupyter Notebook* tidak dapat dimanfaatkan langsung oleh pengguna akhir (*end-user*). Diperlukan proses *deployment* ke dalam arsitektur web terintegrasi. Proyek ini bertujuan membangun aplikasi berbasis web yang mampu memprediksi estimasi total penjualan global (*Global Sales*) suatu judul *video game* berdasarkan platform, genre, tahun rilis, dan publisher, sekaligus menyimpan rekam jejak simulasi ke dalam database lokal.

### 1.2 Tujuan Utama
1. Mengimplementasikan model *Machine Learning* (Regresi) menggunakan dataset `vgsales.csv` yang tersedia dan membekukannya menggunakan `joblib`; jumlah baris bersih ditentukan lewat proses EDA aktual.
2. Membangun *Backend REST API* menggunakan **Django** untuk memproses permintaan prediksi dan mengelola komunikasi ke basis data.
3. Menyimpan riwayat prediksi dan parameter input pengguna ke SQLite lokal; menyiapkan konfigurasi agar dapat dialihkan ke PostgreSQL pada tahap berikutnya.
4. Membangun antarmuka pengguna (*Frontend*) yang interaktif, responsif, dan mudah dipahami menggunakan **Streamlit**.

---

## 2. Arsitektur Sistem
Aplikasi mengadopsi pola arsitektur tiga lapis (*3-Tier Architecture*) untuk memisahkan antara antarmuka pengguna, logika bisnis/model AI, dan penyimpanan data:

```text
[ Pengguna / Browser ]
         │
         ▼ (Interaksi UI: Form Input & Grafik)
[ Frontend: Streamlit (app.py) ]
         │
         ▼ (HTTP POST / GET Request - JSON)
[ Backend: Django REST API ] ──────► [ Model ML: model_vgsales.joblib ]
         │
         ▼ (Django ORM)
[ Database: SQLite lokal ]
```

---

## 3. Spesifikasi Data & Machine Learning
### 3.1 Dataset (`vgsales.csv`)
* **Sumber Data:** Berkas `dataset/vgsales.csv` tersedia di workspace; ukuran dataset aktual dicatat oleh proses EDA.
* **Pra-pemrosesan (*Preprocessing*):**
  * Mencatat struktur kolom, tipe data, dan missing values.
  * Menghapus hanya duplikat identik seluruh kolom dan baris missing pada `Platform`, `Genre`, `Year`, `Publisher`, dan `Global_Sales`.
  * Mengubah `Year` dan `Global_Sales` menjadi numerik; menghapus nilai numerik tidak valid, non-finite, dan target negatif. Tidak menghapus outlier.
  * Menggunakan `ColumnTransformer` di dalam `Pipeline`: `OneHotEncoder(handle_unknown="ignore")` untuk `Platform`, `Genre`, `Publisher`; `Year` numerik tanpa scaling.

### 3.2 Variabel Input & Target
* **Fitur Input ($X$):**
  1. `Platform` (Kategorikal: PS4, PC, XOne, Switch, PS3, X360, dll.)
  2. `Genre` (Kategorikal: Action, Shooter, Sports, RPG, Racing, Platform, dll.)
  3. `Year` (Numerik: Rentang 1980 – 2016)
  4. `Publisher` (Kategorikal)
* `Name` hanya metadata opsional `game_title`; tidak digunakan sebagai fitur.
* `Rank`, `NA_Sales`, `EU_Sales`, `JP_Sales`, dan `Other_Sales` tidak digunakan sebagai fitur.
* **Target Output ($y$):**
  * `Global_Sales` (Numerik Kontinu: Estimasi total penjualan global dalam satuan juta kopi).

### 3.3 Algoritma & Penyimpanan Model
* **Kandidat Algoritma:** `LinearRegression` dan `RandomForestRegressor`; pembagian data 80:20 dengan `random_state=42`. Preprocessing hanya di-fit pada training set. Kandidat dipilih dengan memprioritaskan RMSE terendah, kemudian R² tertinggi. Laporkan R², RMSE, dan MAE aktual; jangan mengasumsikan performa.
* **Artefak Model:** Disimpan dalam format `model_vgsales.joblib` menggunakan `joblib.dump()`.

---

## 4. Spesifikasi Fungsional (*Functional Requirements*)

### 4.1 Frontend (Streamlit — `app.py`)
| ID | Fitur | Deskripsi Fungsional | Komponen Streamlit |
| :--- | :--- | :--- | :--- |
| **FR-01** | Header & Identitas | Menampilkan judul aplikasi, deskripsi singkat, dan identitas anggota kelompok jika tersedia. | `st.title()`, `st.markdown()` |
| **FR-02** | Form Input Simulasi | Menyediakan judul game opsional, Platform, Genre, Tahun Rilis, dan Publisher. Pilihan dimuat dari endpoint metadata Django. | `st.selectbox()`, `st.number_input()`, `st.text_input()` |
| **FR-03** | Eksekusi Prediksi | Tombol aksi untuk mengirimkan payload JSON input pengguna ke endpoint API Django. | `st.button()` |
| **FR-04** | Tampilan Hasil Prediksi | Menampilkan hasil estimasi `Global_Sales` dalam juta kopi serta kategori kesuksesan game (*Niche*, *Standard*, atau *Blockbuster*). | `st.success()`, `st.metric()` |
| **FR-05** | Visualisasi Data Historis | Menampilkan grafik rata-rata penjualan berdasarkan Genre dan Platform dari dataset hasil preprocessing. | `st.bar_chart()` |
| **FR-06** | Riwayat Prediksi (Log) | Mengambil maksimal 20 riwayat simulasi terbaru dari database lokal melalui Django API. | `st.dataframe()` |

### 4.2 Backend (Django)
| ID | Endpoint / Modul | Deskripsi Fungsional |
| :--- | :--- | :--- |
| **BE-01** | `POST /api/predict/` | Menerima `game_title` opsional, platform, genre, release_year, publisher; memvalidasi payload, memakai Pipeline `model_vgsales.joblib`, membatasi estimasi minimal nol, mengkategorikan dan menyimpan hasil lewat Django ORM. |
| **BE-02** | `GET /api/history/` | Mengambil maksimal 20 riwayat prediksi terbaru dari SQLite untuk ditampilkan di Streamlit. |
| **BE-03** | `GET /api/metadata/` | Mengembalikan pilihan platform, genre, publisher, rentang tahun, dan rata-rata penjualan agregat untuk grafik dari dataset bersih. |

### 4.3 Database (SQLite lokal; PostgreSQL untuk migrasi selanjutnya)
* **Nama Tabel:** `prediction_logs`
* **Skema Kolom:**
  * `id` (Primary Key, auto-increment)
  * `created_at` (waktu pencatatan)
  * `game_title` (opsional)
  * `platform`, `genre`, `publisher` (wajib)
  * `release_year` (wajib)
  * `predicted_global_sales` (wajib, juta kopi)

---

## 5. Spesifikasi Non-Fungsional (*Non-Functional Requirements*)
1. **Kecepatan Inferensi:** Model `.joblib` dimuat sekali per proses Django dan digunakan kembali untuk request berikutnya.
2. **Penanganan Error (*Error Handling*):** Jika input di luar rentang atau API Django tidak tersedia, Streamlit menampilkan pesan ramah pengguna (`st.error()`) tanpa membuat aplikasi *crash*.
3. **Keterbacaan Kode & Dokumentasi:** Struktur folder proyek harus rapi dan dilengkapi dengan file `README.md` berisi cara instalasi, arsitektur sistem, sumber dataset, serta pembagian tugas kelompok sesuai rubrik penilaian Pertemuan 14.

---

## 6. Struktur Direktori Proyek yang Disarankan
```text
vgsales-intelligent-system/
├── dataset/
│   └── vgsales.csv                  # Dataset mentah
├── ml/
│   └── train_model.py               # EDA, cleaning, preprocessing, training, evaluasi
├── artifacts/
│   ├── model_vgsales.joblib
│   ├── model_metadata.json
│   └── cleaned_vgsales.csv
├── reports/
│   └── eda_summary.md
├── backend_django/
│   ├── manage.py
│   ├── config/                      # Pengaturan proyek Django
│   └── predictor/                   # App Django (views.py, urls.py, model_vgsales.joblib)
├── frontend_streamlit/
│   └── app.py                       # Kode utama antarmuka Streamlit
├── requirements.txt                 # Daftar library (Django, Streamlit, scikit-learn, joblib, pandas)
└── README.md                        # Dokumentasi lengkap proyek & pembagian tugas
```

---

## 7. Tahapan Eksekusi (*Milestones*)
1. **Fase 1 (Pelatihan Model):** Pembersihan data `vgsales.csv`, pembuatan *Pipeline* Scikit-Learn, pelatihan model regresi, dan ekspor file `model_vgsales.joblib`.
2. **Fase 2 (Setup Database):** Membuat tabel lokal `prediction_logs` melalui Django ORM; tanpa koneksi Supabase.
3. **Fase 3 (Pengembangan Backend):** Inisialisasi proyek Django, konfigurasi SQLite (dengan opsi konfigurasi PostgreSQL), serta endpoint `/api/predict/`, `/api/history/`, dan `/api/metadata/`.
4. **Fase 4 (Pengembangan Frontend):** Pembuatan halaman interaktif Streamlit (`app.py`), integrasi `requests` ke API Django, dan penambahan grafik visualisasi data.
5. **Fase 5 (Pengujian & Dokumentasi):** Uji coba alur lokal (input Streamlit → Django → SQLite → riwayat Streamlit) dan finalisasi `README.md`.