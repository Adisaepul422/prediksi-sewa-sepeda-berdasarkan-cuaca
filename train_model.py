"""
Melatih model Regresi Linear Sederhana: cnt (penyewaan sepeda per hari) ~ temp.

Langkahnya sama dengan notebook Colab (TA02_Regresi_Linear_Sepeda.ipynb):
bagi data 80/20 dengan random_state=42, latih LinearRegression, hitung metrik,
lalu simpan model dengan Joblib dan ringkasannya ke JSON.

Jalankan:  python train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "day.csv")
MODEL_PATH = os.path.join(BASE, "model", "model_regresi_sepeda.joblib")
SUMMARY_PATH = os.path.join(BASE, "model", "ringkasan_model.json")


def train(data_path=DATA_PATH, model_path=MODEL_PATH, summary_path=SUMMARY_PATH):
    df = pd.read_csv(data_path)
    X = df[["temp"]]
    y = df["cnt"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    ringkasan = {
        "jumlah_data": int(len(df)),
        "korelasi_temp_cnt": float(df["temp"].corr(df["cnt"])),
        "beta0": float(model.intercept_),
        "beta1": float(model.coef_[0]),
        "R2_uji": float(r2_score(y_test, pred)),
        "RMSE_uji": float(np.sqrt(mean_squared_error(y_test, pred))),
        "temp_min": float(df["temp"].min()),
        "temp_max": float(df["temp"].max()),
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, indent=2)
    return model, ringkasan


if __name__ == "__main__":
    _, hasil = train()
    print("Model disimpan di:", MODEL_PATH)
    print(json.dumps(hasil, indent=2))
