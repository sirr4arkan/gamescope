# Video Game Global Sales Prediction

Aplikasi lokal untuk memperkirakan `Global_Sales` sebuah video game sebelum
peluncuran, menggunakan informasi platform, genre, tahun, dan publisher.
Prediksi dinyatakan dalam juta unit terjual dan merupakan estimasi pola historis, bukan
jaminan penjualan aktual.

## Tujuan dan alur

`dataset/vgsales.csv` → pembersihan → preprocessing dan training Pipeline →
evaluasi → model joblib → Django REST API → Supabase PostgreSQL → Streamlit.

Pipeline encoding berada bersama estimator di dalam satu artefak sehingga
encoding training dan prediksi konsisten. Django memuat Pipeline saat pertama
kali dipakai, menyimpannya dalam cache proses, memvalidasi payload, membatasi
prediksi minimum nol, mengategorikan hasil, lalu menyimpan request ke PostgreSQL.
Streamlit meminta pilihan formulir, grafik historis, prediksi, dan riwayat
melalui Django API; Streamlit tidak terhubung langsung ke Supabase. Halaman
Model AI menampilkan materi edukatif dan dua diagram
statis. Penjelasan AI untuk grafik Genre dan Platform tersedia di halaman
Analisis Pasar dan hanya dibuat setelah tombol terkait diklik; backend mengambil
agregat aktual lalu meminta jawaban Markdown dari Groq. API key tidak pernah
dikirim ke Streamlit.

## Dataset dan variabel

Sumber data adalah berkas asli `dataset/vgsales.csv` yang tersedia bersama
workspace. PRD yang disediakan menyebutnya sebagai Open Dataset Video Game
Sales, tetapi tidak memberikan URL sumber atau informasi lisensi. Karena itu,
lisensi dan atribusi eksternal belum dapat diverifikasi. Pastikan hak
redistribusi dataset sebelum menerbitkan atau membagikan proyek.

- **X**: `Platform`, `Genre`, `Year`, `Publisher`.
- **y**: `Global_Sales` (juta unit terjual).
- `Name` hanya metadata opsional `game_title`; tidak masuk ke fitur.
- `Rank`, `NA_Sales`, `EU_Sales`, `JP_Sales`, dan `Other_Sales` tidak digunakan
  sebagai fitur.

Cleaning membuang hanya duplikat identik berdasarkan seluruh kolom, missing
value pada empat fitur dan target, nilai Year/Global_Sales yang tidak numerik
atau tidak finite, serta target negatif. Outlier tidak dihapus. Jumlah baris
dan ringkasan struktur sumber ditulis ke `artifacts/model_metadata.json` dan
`reports/eda_summary.md` setiap kali training.

Fitur kategorikal diubah dengan `OneHotEncoder(handle_unknown="ignore")` dalam
`ColumnTransformer`; `Year` diteruskan apa adanya tanpa scaling. Dua kandidat
adalah `LinearRegression` dan `RandomForestRegressor` dengan 200 estimator.
Data dibagi 80:20 dengan `random_state=42`; preprocessing di-fit hanya melalui
training set karena berada di dalam `Pipeline`. Pemilihan model mengutamakan
RMSE terendah, kemudian R² tertinggi jika RMSE sama. Metrik aktual kedua model
dicatat dalam `artifacts/model_metadata.json` setelah training—tidak
ditetapkan sebelumnya.
Metrik `evaluations` mengukur output mentah estimator dan dipakai untuk
memilih model; `evaluations_after_nonnegative_clipping` mengukur output sesudah
batas minimum nol yang sama dengan respons API. Clipping dapat mengubah metrik,
jadi keduanya dilaporkan terpisah.

## Hasil aktual pada dataset workspace

Training berhasil pada 16.324 baris sumber: 0 duplikat identik, 36 baris
dihapus karena `Publisher` missing, dan 16.288 baris tersisa. Nilai numerik
invalid dan target negatif/non-finite tidak ditemukan. Test set berisi 3.258
baris.

| Model | R² | RMSE (juta unit terjual) | MAE (juta unit terjual) |
|---|---:|---:|---:|
| LinearRegression | 0.093313 | 1.892897 | 0.559662 |
| RandomForestRegressor | -0.023645 | 2.011282 | 0.531978 |

`LinearRegression` dipilih karena RMSE lebih rendah dan R² lebih tinggi,
meskipun MAE Random Forest sedikit lebih rendah. Hasil ini mencerminkan
baseline fitur yang tersedia dan tidak menjamin akurasi pada game tertentu.
Untuk Linear Regression, setelah prediksi negatif di-clamp ke nol seperti
perilaku API, metriknya adalah R² 0.095198, RMSE 1.890928, dan MAE 0.542939.
Metrik mentah untuk pemilihan model dan metrik setelah clipping disimpan
terpisah dalam metadata. Evaluasi tambahan menunjukkan hasil temporal yang
lemah; data training berakhir pada 2016 dan model sangat meremehkan game
blockbuster.

## Arsitektur dan struktur

```text
dataset/vgsales.csv
ml/train_model.py
artifacts/
  cleaned_vgsales.csv
  model_metadata.json
  model_vgsales.joblib
reports/eda_summary.md
backend_django/
  manage.py
  config/
  predictor/
frontend_streamlit/app.py
frontend_streamlit/pages/
  home.py
  prediction.py
  analysis.py
  model.py
  history.py
  guide.py
frontend_streamlit/ui/
  api_client.py
  charts.py
  components.py
  diagrams.py
  styles.css
frontend_streamlit/assets/hero-forest.svg
.streamlit/config.toml
requirements.txt
```

Frontend Streamlit menggunakan navigasi enam halaman: **Beranda**, **Prediksi**,
**Analisis Pasar**, **Model AI**, **Riwayat**, dan **Panduan**. Tema nature-inspired
menggunakan light theme Streamlit, CSS lokal, dan ilustrasi SVG yang disertakan.
Latar luar Warm Ivory membingkai panel konten Soft Sage terpusat dengan sudut
rounded; section alternatif dan card putih memakai border sage lembut. Beranda
menampilkan tiga fitur inti serta alur Karakteristik Rilis → Prediksi Machine
Learning → Estimasi Penjualan, tanpa mengubah alur prediksi.
Grafik pasar menggunakan agregat aktual dengan warna serta axis terang; halaman
Model AI menampilkan alur ML dan diagram sistem yang sudah ada dan tidak
diubah. Seluruh opsi form, agregat grafik, estimasi, dan riwayat tetap
dimuat melalui backend Django.

Database aktif dikonfigurasi melalui `backend_django/.env` dan menggunakan
Supabase PostgreSQL (`DB_ENGINE=postgresql`). Django ORM mengelola tabel
`prediction_logs`; database SQLite lokal tetap dipertahankan sebagai sumber
arsip dan tidak dihapus. Frontend hanya berkomunikasi dengan Django REST API.
Koneksi PostgreSQL memakai psycopg dan TLS `DB_SSLMODE=require`.

## Instalasi

Gunakan Python 3.10 atau lebih baru. Dari direktori proyek di PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika PowerShell melarang aktivasi skrip, jalankan interpreter dan executable
secara langsung, misalnya `.\.venv\Scripts\python.exe`.

## Training model

```powershell
.\.venv\Scripts\python.exe .\ml\train_model.py
```

Perintah ini membaca dataset asli, menghasilkan CSV bersih, Pipeline terpilih,
metadata metrik aktual, dan laporan EDA. Training ulang akan memperbarui
artefak tersebut.

## Menjalankan aplikasi lokal

### Konfigurasi database

Salin `backend_django/.env.example` ke `backend_django/.env` bila belum ada,
lalu isi `DB_USER`, `DB_PASSWORD`, dan `DB_HOST` dari pengaturan koneksi
PostgreSQL Supabase. Jangan commit `.env` atau menaruh kredensial database di
frontend. Konfigurasi Groq tetap berada di `.env` backend yang sama.

Sebelum migrasi awal, buat salinan database SQLite lokal; file aslinya tetap
dipertahankan:

```powershell
New-Item -ItemType Directory -Force .\backend_django\backups | Out-Null
Copy-Item .\backend_django\db.sqlite3 ('.\backend_django\backups\db.sqlite3-' + (Get-Date -Format yyyyMMdd-HHmmss) + '.bak')
```

Terapkan migration Django, periksa koneksi dan operasi insert/select, lalu
impor histori lama. Perintah impor aman dijalankan ulang:

```powershell
.\.venv\Scripts\python.exe .\backend_django\manage.py migrate
.\.venv\Scripts\python.exe .\backend_django\manage.py check_database
.\.venv\Scripts\python.exe .\backend_django\manage.py migrate_sqlite_history
```

1. Jalankan backend:

   ```powershell
   .\.venv\Scripts\python.exe .\backend_django\manage.py runserver 127.0.0.1:8000 --insecure
   ```

   Karena konfigurasi lokal mempertahankan `DJANGO_DEBUG=false`,
   `--insecure` diperlukan agar Django `runserver` menyajikan static files
   melalui staticfiles finders. Gunakan opsi ini hanya untuk pengembangan lokal;
   untuk deployment, jalankan `collectstatic` dan sajikan aset melalui web
   server/CDN.

2. Pada terminal lain, jalankan frontend:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run .\frontend_streamlit\app.py
   ```

   Buka URL lokal Streamlit yang ditampilkan. URL API default adalah
   `http://127.0.0.1:8000/api`; dapat diganti melalui `DJANGO_API_URL`.

### Django Admin

Panel administrasi tersedia di `http://127.0.0.1:8000/admin/` dan memakai
database Supabase PostgreSQL yang sama. Akses hanya diberikan kepada superuser
atau staff dengan izin model yang sesuai; tidak ada pendaftaran administrator
publik. Buat akun administrator secara interaktif jika belum ada:

```powershell
.\.venv\Scripts\python.exe .\backend_django\manage.py createsuperuser
```

Django meminta username, email, dan password di terminal, lalu menyimpan
password menggunakan hashing bawaan. Selama pemeriksaan integrasi tidak ada
superuser yang terdaftar, jadi jalankan perintah tersebut sebelum login.
Histori prediksi dikelola di kategori **Prediction Management**; akun dan grup
berada di **Authentication & Authorization**. Record dapat dihapus satu per
satu atau secara massal melalui action admin, selalu dengan halaman konfirmasi
dan hanya bagi akun yang memiliki izin delete.

### Endpoint

| Method | Path | Keterangan |
|---|---|---|
| POST | `/api/predict/` | Prediksi dan simpan riwayat |
| GET | `/api/history/` | Maksimal 20 riwayat terbaru |
| GET | `/api/metadata/` | Pilihan form, tahun, dan agregat grafik |
| POST | `/api/explain/` | Penjelasan AI untuk analisis grafik yang diizinkan |

Payload prediksi menerima `platform`, `genre`, `release_year`, `publisher`,
serta `game_title` opsional. Kesalahan validasi dikembalikan sebagai JSON
dengan status 400; artefak metadata/model yang tidak tersedia menghasilkan
status 503.

### Penjelasan grafik pasar dengan AI

Isi `GROQ_API_KEY` di `backend_django/.env` dengan key milik backend dan, bila
perlu, pilih `GROQ_MODEL`. Django memuat berkas tersebut pada proses backend
saja. Model default `openai/gpt-oss-20b`;
provider dan model dapat berubah ketersediaannya, sehingga periksa daftar model
produksi Groq jika permintaan gagal. Jangan menaruh key di frontend, commit
berkas `.env`, atau mengirimkannya melalui request browser.

Di halaman **Analisis Pasar**, gunakan tombol penjelasan di bawah grafik Genre
atau Platform. Jawaban masing-masing grafik disimpan terpisah selama sesi dan
tombol yang sama dapat digunakan untuk menghasilkan ulang. Endpoint hanya
menerima `diagram_type` bernilai `genre_analysis` atau `platform_analysis`;
prompt ditentukan server. Backend mengambil rata-rata Global Sales, jumlah
observasi per kategori, total baris, dan rentang tahun dari dataset bersih yang
sama dengan agregat grafik. Tanpa key, grafik tetap tampil dan aplikasi
menampilkan pesan bahwa fitur AI belum dikonfigurasi.

## Pengujian

Setelah training, jalankan:

```powershell
.\.venv\Scripts\python.exe .\backend_django\manage.py test predictor
.\.venv\Scripts\python.exe .\frontend_streamlit\test_app.py
```

Tes mencakup endpoint metadata, prediksi dengan pencatatan database,
penanganan input invalid/model/database, history kosong dan batas 20 item,
pembersihan data, pemuatan Pipeline joblib sekali, serta prediksi kategori
yang belum pernah dilihat. Training aktual juga menguji ekspor ulang
`LinearRegression` dan pencatatan metrik sebelum/sesudah clipping. Pemeriksaan
Integrasi Streamlit dilakukan dengan tes AppTest pada halaman multipage,
termasuk render grafik, hasil/riwayat, penjelasan grafik hanya setelah tombol
ditekan, isolasi jawaban genre/platform, generate ulang, serta penanganan
backend/provider yang tidak tersedia.
Pengujian backend untuk fitur Groq menggunakan mock; permintaan live hanya
tersedia setelah `GROQ_API_KEY` disiapkan.

## Pembagian tugas

Modul 14 menetapkan kelompok beranggotakan 3–4 orang dan meminta pembagian
tugas dicatat. Nama anggota dan pembagian tugas kelompok ini tidak tersedia
di PRD/workspace, sehingga tidak ditambahkan dengan asumsi.
