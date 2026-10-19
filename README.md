# Prediksi Jumlah Penyewaan Sepeda Berdasarkan Suhu Udara

Tugas Akhir 2 (TA-2) Praktikum Pembelajaran Mesin. Aplikasi web Flask yang memakai model
Regresi Linear Sederhana untuk memprediksi jumlah penyewaan sepeda per hari dari suhu udara.

- **Variabel X:** `temp` (suhu, dinormalisasi: suhu °C ÷ 41)
- **Variabel y:** `cnt` (total penyewaan sepeda per hari)
- **Dataset:** [Bike Sharing Dataset](https://www.kaggle.com/marklvl/bike-sharing-dataset), file harian `day.csv` (731 hari)
- **Hasil pada data uji:** R² = 0,40, RMSE = 1.546 sepeda per hari
- **Persamaan:** `cnt = 1272,16 + 6575,39 × temp`

## Struktur proyek

```
bike-rental-flask/
├── app.py                 # backend Flask
├── train_model.py         # melatih ulang model dari data/day.csv
├── requirements.txt
├── Procfile
├── data/day.csv
├── model/
│   ├── model_regresi_sepeda.joblib
│   └── ringkasan_model.json
├── templates/index.html   # antarmuka (Bootstrap 5)
└── static/img/            # grafik dari notebook
```

## Menjalankan di komputer (VS Code)

Buka folder proyek di VS Code, lalu di terminal:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000 di browser.

Catatan: jika model `.joblib` tidak bisa dimuat (misalnya beda versi scikit-learn), aplikasi otomatis
melatih ulang model dari `data/day.csv`. Anda juga bisa melatih manual dengan `python train_model.py`.

## Memakai model dari notebook Colab sendiri

Notebook Colab menghasilkan `model_regresi_sepeda.joblib` dan `ringkasan_model.json`.
Salin keduanya ke folder `model/` untuk menggantikan file yang ada. Hasilnya sama selama
memakai `random_state=42` dan pembagian data 80/20.

## Endpoint

| Alamat | Fungsi |
| --- | --- |
| `/` | Halaman web dengan form prediksi |
| `/api/predict?suhu=28` | Prediksi dalam format JSON |
| `/health` | Pengecekan status |

## Unggah ke GitHub

```bash
git init
git add .
git commit -m "Aplikasi Flask prediksi penyewaan sepeda"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```

Buat repositori kosong dulu di github.com (tanpa README), lalu ganti `USERNAME` dan `NAMA-REPO`.

## Deploy ke Render (gratis)

1. Buka render.com, login dengan akun GitHub.
2. Pilih **New → Web Service**, lalu pilih repositori ini.
3. Isi: **Runtime** Python 3, **Build Command** `pip install -r requirements.txt`,
   **Start Command** `gunicorn app:app`, **Instance Type** Free.
4. Klik **Create Web Service** dan tunggu build selesai. Tautan aplikasi muncul di bagian atas halaman.

Pada paket gratis, aplikasi tidur saat lama tidak diakses, jadi akses pertama bisa lambat.

## Sitasi dataset

Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge",
*Progress in Artificial Intelligence* (2013): pp. 1-15, Springer Berlin Heidelberg,
doi:10.1007/s13748-013-0040-3.
# prediksi-sewa-sepeda-berdasarkan-cuaca
