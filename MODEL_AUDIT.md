# Audit ML dan Aplikasi Prediksi Penjualan Video Game

**Tanggal audit:** 18 November 2026  
**Ruang lingkup:** dataset workspace, kode ML/Django/Streamlit, artefak tersimpan,
tes, evaluasi ulang, dan isi paket eksternal.

## 1. Ringkasan

Training ulang pada CSV asli berhasil dan mereproduksi metrik model serta
metadata sebelumnya (selisih floating-point maksimum sekitar `4.4e-16`).
`LinearRegression` tetap menjadi model yang disimpan karena paling baik pada
random holdout yang ditetapkan, tetapi keuntungan prediktifnya kecil dan
performa temporal mendekati baseline rata-rata. Kinerja sangat lemah untuk
game berpenjualan tinggi. Model layak dipertahankan sebagai baseline
demonstrasi, bukan sebagai estimasi penjualan yang andal.

**Status:** preprocessing dan alur aplikasi konsisten dengan fitur yang
diminta; mutu prediksi terbatas. Tidak ada source code aplikasi atau artefak
model utama yang diubah saat audit awal. Finalisasi sesudahnya membuat
perubahan terbatas pada penanganan error/cache, perhitungan metrik
pasca-clipping, dokumentasi, dan tes; estimator utama tetap
`LinearRegression`. File diagnostik JSON dan ZIP juga diperbarui.

## 2. Dataset dan kualitas data — temuan terbukti

Dataset `dataset/vgsales.csv` berisi 16.324 baris dan 11 kolom. Tipe sumber:
`Rank`, `Year` integer; `Name`, `Platform`, `Genre`, `Publisher` object; kolom
penjualan regional dan `Global_Sales` float.

| Pemeriksaan | Hasil |
|---|---:|
| Duplikat identik seluruh kolom | 0 |
| Missing `Publisher` | 36 |
| Missing pada kolom lainnya | 0 |
| Baris setelah cleaning | 16.288 |
| Rentang tahun bersih | 1980–2016 |
| Platform / Genre / Publisher unik | 31 / 12 / 576 |
| Publisher yang hanya muncul sekali | 192 |

Target tidak negatif, dengan rentang 0,01–82,74 juta kopi. Median 0,17,
rata-rata 0,541, dan skewness 17,303. Sebanyak 87,36% data bersih berada di
bawah 1 juta kopi dan 1,26% mencapai setidaknya 5 juta kopi. Kuartil 99 dan
99,5 masing-masing 5,473 dan 8,291 juta. Batas atas aturan IQR adalah 1,11;
1.826 baris melampauinya. Ini penanda statistik untuk pemeriksaan, bukan bukti
bahwa data tersebut salah, dan cleaning tidak menghapusnya.

Distribusi fitur tidak merata. Platform terbanyak adalah DS (2.131) dan PS2
(2.127); genre terbanyak Action (3.250); tahun dengan observasi terbanyak
adalah 2009 (1.433) dan 2008 (1.428). Publisher memiliki banyak kategori
langka. Rincian sumber dan audit pembersihan dapat dilihat di
`reports/eda_summary.md`; tabel diagnostik lengkap ada di
`reports/model_audit_metrics.json`.

## 3. Preprocessing dan Pipeline — temuan terbukti

- Cleaning menyalin input, menghapus duplikat identik seluruh kolom, membuang
  missing pada fitur/target wajib, mengonversi `Year` dan `Global_Sales` ke
  numerik, lalu membuang nilai numerik non-finite serta target negatif.
- Pada CSV ini, 36 baris yang dibuang memiliki `Publisher` kosong; tidak ada
  nilai numerik invalid atau target negatif/non-finite. Outlier dipertahankan.
- Fitur yang benar-benar digunakan: `Platform`, `Genre`, `Year`, `Publisher`;
  target `Global_Sales`. `Name`, `Rank`, dan kolom regional tidak masuk ke X.
- Artefak joblib berhasil dimuat. Isinya `Pipeline` dengan
  `ColumnTransformer`, `OneHotEncoder(handle_unknown="ignore")` untuk tiga
  kolom kategorikal, passthrough untuk `Year`, lalu `LinearRegression`.
  `feature_names_in_` cocok dengan metadata. Pipeline yang sama digunakan
  untuk inference, jadi encoder tidak di-fit ulang pada request.
- Tidak ditemukan target leakage dari kolom yang dilarang. Random holdout
  bukan pemisahan berdasarkan game/judul, sehingga port/varian judul terkait
  dapat tersebar di train dan test. Dampak terhadap estimasi generalisasi
  belum diuji dengan Group split.

## 4. Training dan validasi metadata — temuan terbukti

Split evaluasi awal 80:20 dengan `random_state=42` menghasilkan 13.030 baris
training dan 3.258 baris test. Pipeline di-fit hanya melalui training set.
Kriteria pemilihan mengutamakan RMSE terendah, dengan R² sebagai pemecah seri.
Evaluasi ulang di lingkungan audit (`scikit-learn 1.9.1`) mereproduksi metrik
metadata, dan model di dalam joblib cocok dengan `selected_model` serta fitur
metadata.

Metadata tidak mencatat versi Python/pandas/scikit-learn/joblib atau hash
artefak. Jadi, kecocokan isi saat audit telah diperiksa, tetapi identitas
lingkungan/artifact training asli tidak dapat dibuktikan secara independen
hanya dari metadata. Joblib/pickle dapat sensitif terhadap perubahan versi;
versi yang diuji saat audit tidak otomatis menjamin kompatibilitas dengan
semua versi yang diizinkan rentang requirements.

Pada finalisasi, training menambahkan `evaluations_after_nonnegative_clipping`
ke metadata. Metrik mentah tetap digunakan untuk seleksi agar proses pemilihan
sebelumnya tidak berubah; metrik pasca-clipping kini ikut tersedia untuk
membandingkan perilaku output API.

## 5. Evaluasi model random holdout

Semua metrik berikut dihitung ulang dari pipeline aktual dengan split
80:20/42 yang sama; skor estimator adalah sebelum clipping hasil API ke nol.

| Model | R² | RMSE (juta) | MAE (juta) |
|---|---:|---:|---:|
| LinearRegression | 0,093313 | 1,892897 | 0,559662 |
| RandomForestRegressor (200 trees) | -0,023645 | 2,011282 | 0,531978 |
| DummyRegressor (mean) | -0,000359 | 1,988273 | 0,617337 |

Linear Regression mengungguli dummy pada RMSE sekitar 4,8% dan MAE sekitar
9,3%, tetapi hanya menjelaskan sekitar 9,3% variasi target. Random Forest
memiliki MAE lebih rendah daripada dummy, namun RMSE dan R² lebih buruk.
Karena penjualan sangat skew, beberapa error besar sangat berpengaruh pada
RMSE/R², sementara MAE dapat memberi urutan yang berbeda.

### 5.1 Cross-validation 5-fold

`KFold(5, shuffle=True, random_state=42)` dijalankan untuk kedua kandidat dan
DummyRegressor. Pipeline di-fit ulang per fold. Nilai berikut adalah mean
beserta standar deviasi antar-fold:

| Model | R² mean ± SD | RMSE mean ± SD (juta) | MAE mean ± SD (juta) |
|---|---:|---:|---:|
| LinearRegression | 0,13565 ± 0,02904 | 1,44432 ± 0,25740 | 0,53661 ± 0,01661 |
| RandomForestRegressor | -0,07213 ± 0,19306 | 1,60124 ± 0,28525 | 0,50484 ± 0,01695 |
| DummyRegressor (mean) | -0,00037 ± 0,00037 | 1,55129 ± 0,25147 | 0,59541 ± 0,02001 |

Cross-validation tetap acak per baris, bukan evaluasi pada judul baru atau
masa mendatang. Skor R² Random Forest negatif dan variasinya lebih besar.

### 5.2 Evaluasi temporal

Sebagai uji tambahan, seluruh data sampai 2011 menjadi training (13.551 baris)
dan tahun 2012–2016 menjadi test (2.737 baris). Ini holdout berdasarkan tahun,
bukan pengganti metrik random split yang disyaratkan.

| Model | R² | RMSE (juta) | MAE (juta) |
|---|---:|---:|---:|
| LinearRegression | 0,004475 | 1,254960 | 0,577343 |
| RandomForestRegressor | 0,054524 | 1,223007 | 0,496322 |
| DummyRegressor (mean) | -0,000718 | 1,258229 | 0,612809 |

Pada periode ini Linear Regression hanya sedikit lebih baik daripada dummy
(RMSE sekitar 0,26% lebih rendah). Random Forest lebih baik pada ketiga metrik
temporal, tetapi tetap memiliki R² rendah. Hasil ini tidak digunakan untuk
menimpa model utama; hasil holdout temporal tunggal juga memerlukan validasi
lebih lanjut.

## 6. Residual dan segmentasi — temuan terbukti

Untuk Linear Regression pada random test set, aktual rata-rata 0,571 juta
dan median 0,18 juta. Residual didefinisikan `aktual - prediksi`: rata-rata
0,033, median -0,118, simpangan baku 1,893. Prediksi mentah berada pada
-0,647 hingga 3,860; **425 dari 3.258** prediksi (13,05%) negatif.

| Segmen berdasarkan aktual | Jumlah | Bias prediksi−aktual (juta) | MAE (juta) | RMSE (juta) |
|---|---:|---:|---:|---:|
| Niche (<1) | 2.833 | +0,228 | 0,346 | 0,506 |
| Standard (1–<5) | 382 | -0,873 | 1,108 | 1,419 |
| Blockbuster (≥5) | 43 | -9,769 | 9,769 | 15,386 |

Model cenderung melebihkan penjualan niche dan sangat meremehkan penjualan
blockbuster. Output API membatasi prediksi minimum nol. Metrik Linear Regression setelah
clipping pada random holdout adalah R² 0,095198, RMSE 1,890928, MAE 0,542939.
Training sekarang menyimpan metrik mentah dan pasca-clipping secara terpisah;
seleksi model tetap berdasarkan metrik mentah.

## 7. Audit source code aplikasi — temuan terbukti

- **ML:** `ml/train_model.py` berisi cleaning, preprocessing, split, fit,
  evaluasi, pemilihan dan ekspor; joblib menyimpan pipeline terpilih.
- **Django:** URL mencakup predict/history/metadata; prediksi memuat model
  dengan cache proses, memvalidasi field serta kategori/tahun, menyimpan ORM
  model `PredictionLog`, dan mengembalikan kategori dengan batas `<1`,
  `1–<5`, dan `≥5`. History dibatasi 20; metadata dibaca dari cleaned CSV.
- **Streamlit:** mengambil metadata, prediksi, dan history via HTTP Django;
  merender form, dua grafik, tabel riwayat, hasil, serta pesan kegagalan API.
- **Database:** migrasi membuat tabel `prediction_logs`; settings default
  SQLite dan mempunyai cabang PostgreSQL environment. Tidak ada koneksi
  Supabase.
- **Tes finalisasi:** 14 tes `predictor` lulus setelah audit. Cakupan termasuk
  cleaning, unknown category Pipeline, threshold kategori, model artifact
  cached, metadata/predict/history, invalid input, kegagalan model/database,
  riwayat kosong, serta batas 20 item. AppTest memverifikasi dua chart,
  prediksi dan riwayat; ketika backend tidak tersedia, error ramah ditampilkan
  tanpa exception Streamlit.
- **Kestabilan finalisasi:** error load model, metadata, serta query/tulis
  database kini dicatat di log server dan dikembalikan sebagai pesan aman tanpa
  detail exception. Frontend tidak lagi menampilkan URL/status atau detail
  koneksi; riwayat memakai cache singkat dan invalidasi setelah prediksi
  berhasil.
- Tidak ada file `.env` di root dan pencarian credential berpola di source
  Python tidak menemukan API key/password/token. `.env.example` berisi
  placeholder. Namun settings mempunyai fallback
  `local-development-only-change-me` untuk `SECRET_KEY`; itu bukan credential
  yang ditemukan, tetapi nilai fallback tidak layak untuk server yang
  terekspos.

### Temuan source yang perlu ditindaklanjuti

1. **Batas tahun API (terbukti):** metadata dan validasi API membatasi input
   tahun ke rentang historis 1980–2016. Aplikasi tidak menerima tahun rilis
   setelah 2016, sehingga penggunaan untuk prediksi tahun berjalan tidak
   tersedia meskipun Year numerik dapat diekstrapolasi. Memperluas batas tidak
   dengan sendirinya membuat prediksi masa depan valid; model harus dilatih
   dan dievaluasi pada data lebih mutakhir dahulu.
2. **Perbedaan clipping/evaluasi (terbukti dan ditangani):** model selection
   memakai skor estimator mentah; API memotong output negatif ke nol.
   Metadata kini menyimpan metrik mentah dan pasca-clipping secara terpisah.
3. **Kategori tak dikenal (terbukti):** OHE mengabaikan kategori baru, tetapi
   API menolak platform/genre/publisher di luar daftar cleaned dataset.
   Frontend dinamis sesuai dengan API saat ini; toleransi unknown kategori
   hanya berlaku jika Pipeline dipanggil langsung.
4. **Keamanan API lokal (terbukti):** endpoint predict ditandai
   `csrf_exempt` dan tidak memakai autentikasi/otorisasi. Ini bukan masalah
   alur lokal yang diuji, tetapi endpoint tidak boleh diekspos ke jaringan
   tanpa desain perlindungan yang sesuai.
5. **Konfigurasi secret (terbukti):** fallback Django secret adalah nilai
   statis. Untuk deployment, minta secret eksplisit dan gagalkan startup
   jika nilainya masih default.
6. **Reproduksibilitas artefak (terbukti):** metadata tidak menyimpan versi
   library, hash dataset/model, atau identitas training code. Lengkapi pada
   training berikutnya agar artifact lineage dapat diaudit; source tidak
   diubah dalam audit ini.

## 8. Interpretasi, hipotesis, dan rekomendasi

### Temuan yang terbukti

- Distribusi target sangat right-skewed dan publisher memiliki kardinalitas
  serta kategori singleton yang tinggi.
- Random split mengunggulkan Linear Regression pada RMSE/R², tetapi temporal
  holdout memberi keuntungan hampir nol dibanding dummy; Random Forest lebih
  baik pada temporal holdout ini.
- Model yang tersedia buruk untuk memprediksi blockbuster dan dipotong ke
  nol oleh API pada prediksi mentah negatif.

### Dugaan yang masih perlu diuji

- Publisher singleton dan data jarang dapat membuat estimasi OHE tidak stabil
  pada kategori yang sedikit terlihat; pengaruhnya perlu diuji dengan split
  berdasarkan judul/publisher dan validasi berulang.
- Distribusi penjualan berubah antar-tahun dan berkontribusi pada lemahnya
  temporal score. Satu cutoff 2011 saja belum membuktikan pola drift secara
  umum.
- Transformasi target `log1p`, pelaporan metrik per kelompok, atau kelompok
  publisher yang jarang mungkin memperbaiki error tipikal, tetapi dapat
  mengubah kualitas prediksi ekor atas dan memerlukan evaluasi balik ke satuan
  juta kopi.

### Rekomendasi pengembangan

1. Pertahankan Linear Regression sebagai baseline arsip, bukan sebagai klaim
   prediksi yang kuat. Jangan mengganti artefak sebelum evaluasi temporal dan
   per-segmen yang terulang pada data lebih baru.
2. Tambahkan evaluasi berbasis kelompok judul untuk memperkirakan performa
   pada game yang benar-benar belum pernah terlihat, serta rolling/temporal
   validation jika data tahun terbaru tersedia.
3. Prioritaskan pengumpulan data pra-rilis yang sah dan lebih mutakhir.
   Regional sales tidak boleh digunakan bila prediksi memang harus dibuat
   sebelum peluncuran.
4. Pertimbangkan transformasi target atau penanganan kategori langka hanya
   sebagai eksperimen terpisah; ukur RMSE, MAE, R², clipping rate, dan metrik
   tiap segmen setelah dikembalikan ke skala asli.
5. Pada training berikutnya simpan versi dependencies, split/CV config,
   hash dataset dan model, serta metrik output setelah post-processing.
6. Sebelum layanan di luar komputer lokal, hilangkan secret fallback,
   tentukan autentikasi/CSRF policy, dan tinjau batas input serta logging.

## 9. Kesimpulan kelayakan

Secara implementasi, pemilihan fitur, cleaning, preprocessing Pipeline,
metadata, artefak, dan integrasi frontend/backend konsisten serta dapat
ditelusuri. Nilai evaluasi yang sebelumnya dilaporkan valid dan direproduksi.
Namun model hanya layak untuk demonstrasi/baseline pembelajaran: Random
holdout R² rendah, temporal Linear Regression nyaris tidak melampaui baseline
rata-rata, tahun data maksimum 2016, dan error blockbuster besar. Tidak
direkomendasikan sebagai jaminan atau dasar komersial penjualan.

## 10. Berkas audit dan batas paket

Hasil numerik lengkap audit tersedia pada `reports/model_audit_metrics.json`;
hasil training final juga mencatat metrik clipping pada
`artifacts/model_metadata.json`.
Paket `vgsales_audit_bundle.zip` memuat source ML/Django/Streamlit, migrations
dan tes, artefak, CSV asli dan cleaned, README, PRD, konfigurasi contoh,
tes AppTest, laporan EDA dan audit. Tidak ada `SOURCE.md` pada workspace;
lisensi dan URL sumber CSV tidak dapat diverifikasi dari dokumen yang
tersedia. Karena tidak diminta sebagai deliverable proyek, PDF Modul Ajar
tidak dimasukkan ke ZIP.
Tidak ada chart raster yang dibuat; distribusi, residual, CV dan temporal
diagnostics dicatat sebagai tabel/angka pada laporan dan JSON.
