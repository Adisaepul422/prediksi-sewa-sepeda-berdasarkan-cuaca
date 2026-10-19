"""
Aplikasi web Flask: Prediksi Jumlah Penyewaan Sepeda Berdasarkan Suhu Udara.

Backend memuat model Regresi Linear (Joblib), menerima suhu dalam °C dari form,
mengubahnya ke skala dataset (temp = suhu / 41), lalu menampilkan prediksi.
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "model_regresi_sepeda.joblib")
SUMMARY_PATH = os.path.join(BASE, "model", "ringkasan_model.json")

# Dokumentasi dataset: kolom temp = suhu (°C) yang dibagi 41 (nilai maksimum).
PEMBAGI_SUHU = 41.0

# Batas input yang diterima form (°C). Di luar batas ini ditolak.
SUHU_MIN_INPUT = -10.0
SUHU_MAX_INPUT = 50.0

app = Flask(__name__)


def muat_model():
    """Muat model dari file Joblib; jika gagal (misal beda versi scikit-learn), latih ulang."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as err:  # noqa: BLE001
        print(f"Model gagal dimuat ({err}). Melatih ulang dari data/day.csv ...")
        from train_model import train

        model, _ = train()
        return model


def muat_ringkasan():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


model = muat_model()
ringkasan = muat_ringkasan()

SUHU_DATA_MIN = ringkasan["temp_min"] * PEMBAGI_SUHU
SUHU_DATA_MAX = ringkasan["temp_max"] * PEMBAGI_SUHU


@app.template_filter("ribuan")
def ribuan(nilai, desimal=0):
    """Format angka gaya Indonesia: 5.217 atau 0,40."""
    teks = f"{nilai:,.{desimal}f}"
    return teks.replace(",", "X").replace(".", ",").replace("X", ".")


def hitung_prediksi(suhu_celsius):
    """Kembalikan dict hasil prediksi untuk suhu (°C)."""
    temp_dataset = suhu_celsius / PEMBAGI_SUHU
    mentah = float(model.predict(pd.DataFrame({"temp": [temp_dataset]}))[0])
    prediksi = max(0.0, mentah)  # jumlah penyewaan tidak mungkin negatif
    rmse = ringkasan["RMSE_uji"]
    return {
        "suhu": suhu_celsius,
        "temp_dataset": temp_dataset,
        "prediksi": round(prediksi),
        "bawah": round(max(0.0, prediksi - rmse)),
        "atas": round(prediksi + rmse),
        "dipotong_nol": mentah < 0,
        "di_luar_rentang": not (SUHU_DATA_MIN <= suhu_celsius <= SUHU_DATA_MAX),
    }


def baca_suhu(teks):
    """Ubah teks input menjadi float; kembalikan (nilai, pesan_error)."""
    if teks is None or str(teks).strip() == "":
        return None, "Suhu wajib diisi."
    try:
        nilai = float(str(teks).strip().replace(",", "."))
    except ValueError:
        return None, "Masukkan suhu berupa angka, misalnya 28 atau 28,5."
    if nilai != nilai or nilai in (float("inf"), float("-inf")):
        return None, "Masukkan suhu berupa angka yang valid."
    if not (SUHU_MIN_INPUT <= nilai <= SUHU_MAX_INPUT):
        return None, f"Suhu harus antara {SUHU_MIN_INPUT:.0f} dan {SUHU_MAX_INPUT:.0f} °C."
    return nilai, None


@app.route("/", methods=["GET", "POST"])
def index():
    hasil, error, nilai_input = None, None, ""
    if request.method == "POST":
        nilai_input = request.form.get("suhu", "")
        suhu, error = baca_suhu(nilai_input)
        if error is None:
            hasil = hitung_prediksi(suhu)
    return render_template(
        "index.html",
        hasil=hasil,
        error=error,
        nilai_input=nilai_input,
        ringkasan=ringkasan,
        suhu_min=SUHU_DATA_MIN,
        suhu_max=SUHU_DATA_MAX,
        pembagi=PEMBAGI_SUHU,
    )


@app.route("/api/predict")
def api_predict():
    """Contoh: /api/predict?suhu=28"""
    suhu, error = baca_suhu(request.args.get("suhu"))
    if error:
        return jsonify({"error": error}), 400
    return jsonify(hitung_prediksi(suhu))


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
