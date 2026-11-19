# EDA dan Pembersihan Dataset VG Sales

- Dataset sumber: `dataset\vgsales.csv`
- Baris mentah: 16,324
- Baris bersih: 16,288
- Kolom: Rank, Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales

## Tipe Data dan Missing Values Sumber

| Kolom | Tipe | Missing |
|---|---|---:|
| Rank | int64 | 0 |
| Name | object | 0 |
| Platform | object | 0 |
| Year | int64 | 0 |
| Genre | object | 0 |
| Publisher | object | 36 |
| NA_Sales | float64 | 0 |
| EU_Sales | float64 | 0 |
| JP_Sales | float64 | 0 |
| Other_Sales | float64 | 0 |
| Global_Sales | float64 | 0 |

## Audit Pembersihan

| Tahap | Jumlah baris |
|---|---:|
| Sebelum pembersihan | 16,324 |
| Duplikat identik dihapus | 0 |
| Missing pada kolom wajib dihapus | 36 |
| Nilai numerik tidak valid dihapus | 0 |
| Target negatif/tidak finite dihapus | 0 |
| Sesudah pembersihan | 16,288 |

Duplikat hanya dihapus jika seluruh kolom identik. Nama game tidak digunakan untuk menghapus baris. Nilai ekstrem tidak dihapus.
