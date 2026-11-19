# GameScope — Video Game Global Sales Prediction

Aplikasi Streamlit untuk memperkirakan `Global_Sales` sebuah video game
menggunakan informasi platform, genre, tahun, dan publisher. Prediksi
dinyatakan dalam juta unit terjual dan merupakan estimasi pola historis,
bukan jaminan penjualan aktual.

## Tujuan dan alur

`dataset/vgsales.csv` → pembersihan → preprocessing dan training Pipeline →
evaluasi → model joblib → inferensi lokal di Streamlit.

Pipeline encoding berada bersama estimator di dalam satu artefak sehingga
encoding training dan prediksi konsisten. Streamlit memuat model, dataset
bersih, dan metadata dari `artifacts/`, melakukan validasi dan inferensi pada
proses server Streamlit, lalu menyimpan/membaca histori melalui Supabase
PostgREST HTTPS. Penjelasan grafik Genre dan Platform dibuat saat diminta
menggunakan Groq HTTPS. Django tetap berada di repository, tetapi bukan
ketergantungan runtime untuk Streamlit Community Cloud.

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
  app_config.py
  charts.py
  components.py
  diagrams.py
  errors.py
  groq_explainer.py
  local_inference.py
  supabase_rest.py
  styles.css
frontend_streamlit/assets/hero-forest.svg
frontend_streamlit/.streamlit/config.toml
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
Model AI menampilkan alur ML dan arsitektur Streamlit, Supabase, dan Groq.
Opsi form, agregat grafik, dan estimasi dihitung dari artifact lokal; riwayat
dibaca dari tabel Supabase yang sama.

Database PostgreSQL Supabase yang sudah ada diakses melalui Data API
(`https://<project-ref>.supabase.co/rest/v1/prediction_logs`), bukan koneksi
PostgreSQL TCP dan bukan Supabase client di browser. Aplikasi menggunakan
publishable/anon key dengan grant dan kebijakan RLS yang sesuai; service-role
key tidak boleh dimasukkan ke Streamlit.

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

Jalankan Streamlit dari root repository:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\frontend_streamlit\app.py
```

Metadata dan prediksi lokal dapat digunakan tanpa Django. Untuk histori dan
penjelasan AI, sediakan konfigurasi melalui environment variables atau
`frontend_streamlit/.streamlit/secrets.toml` (jangan commit file secrets):

```toml
SUPABASE_URL = "https://<project-ref>.supabase.co"
SUPABASE_ANON_KEY = "<publishable-or-legacy-anon-key>"
GROQ_API_KEY = "<groq-api-key>"
GROQ_MODEL = "openai/gpt-oss-20b"
```

`SUPABASE_ANON_KEY` juga dapat berupa publishable key baru yang disimpan
dengan nama `SUPABASE_PUBLISHABLE_KEY`. Jangan gunakan `service_role` key.
Riwayat membutuhkan akses `SELECT` ke `prediction_logs`; penyimpanan prediksi
membutuhkan akses `INSERT` dan sequence yang dipakai kolom ID. Jika RLS aktif,
berikan kebijakan minimum yang sesuai untuk role `anon`. Aplikasi menampilkan
pesan konfigurasi/permission tanpa menampilkan credential atau body error
Supabase.

Penjelasan grafik hanya dibuat setelah tombol Genre atau Platform diklik.
Prompt dibatasi pada agregat dataset aktual, maksimal 450 completion tokens,
dan jawaban maksimal 8.000 karakter. API key Groq dibaca server-side dari
Streamlit Secrets dan tidak dikirim ke browser.

### Django Admin opsional

Kode Django dan panel `/admin/` tetap ada di repository, tetapi tidak diperlukan
untuk menjalankan atau men-deploy frontend Streamlit. Bila Django dijalankan
terpisah untuk administrasi lokal, ikuti konfigurasi backend yang ada dan
gunakan `python backend_django/manage.py createsuperuser` untuk membuat akun
administrator secara interaktif. Streamlit tidak menggunakan `DJANGO_API_URL`.

## Deployment

### Streamlit Community Cloud

Hubungkan repository ke Streamlit Community Cloud dan tetapkan entry point
`frontend_streamlit/app.py`. Pada **Advanced settings → Secrets**, masukkan:

```toml
SUPABASE_URL = "https://<project-ref>.supabase.co"
SUPABASE_ANON_KEY = "<publishable-or-legacy-anon-key>"
GROQ_API_KEY = "<groq-api-key>"
GROQ_MODEL = "openai/gpt-oss-20b"
```

Pastikan folder root `artifacts/` dan dependency `requirements.txt` ikut
ter-deploy. Django hosting, `DJANGO_API_URL`, migrasi database, dan impor
histori tidak diperlukan. Aplikasi memakai Supabase Data API melalui HTTPS;
jika respons menolak akses, periksa Data API exposure, grant, dan kebijakan
RLS untuk role `anon` (SELECT dan INSERT). Jangan menaruh `service_role` key
di secrets aplikasi.

## Pengujian

Setelah training, jalankan:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\frontend_streamlit -p test_app.py
```

Tes menggunakan artifact model lokal yang tersedia, mem-mock seluruh request
Supabase dan Groq, serta memeriksa validasi konfigurasi, prediksi, error
permission, UI multipage, dan penjelasan AI. Tes tidak menghubungi atau menulis
ke database Supabase produksi.

## Pembagian tugas

Modul 14 menetapkan kelompok beranggotakan 3–4 orang dan meminta pembagian
tugas dicatat. Nama anggota dan pembagian tugas kelompok ini tidak tersedia
di PRD/workspace, sehingga tidak ditambahkan dengan asumsi.
