import csv
import os
import random
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, Request

# 1. Inicjalizacja aplikacji FastAPI
app = FastAPI(
    title="Nocarz API",
    description="Mikroserwis do detekcji anomalii cenowych z wbudowanym testem A/B",
)

# 2. Wczytanie zapisanych modeli podczas startu aplikacji
print("Wczytywanie modeli...")
rf_model = joblib.load("generated/random_forest_production.joblib")
iso_model = joblib.load("generated/isolation_forest_baseline.joblib")

LOG_FILE = "ab_test_logs.csv"

# 3. Inicjalizacja pliku z logami testu A/B (jeśli nie istnieje, tworzymy nagłówki)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["timestamp", "offer_id", "model_assigned", "prediction_result"]
        )


# 4. Główny endpoint przyjmujący zapytania (metoda POST)
@app.post("/predict_price_anomaly")
async def predict_price(request: Request):
    # Pobranie surowych danych w formacie JSON
    data = await request.json()

    # Wyciągnięcie ID oferty do logów (jeśli brakuje, wstawiamy "unknown")
    offer_id = data.get("id", "unknown_id")

    # ---------------------------------------------------------
    # DOPASOWANIE STRUKTURY DANYCH DO WYMAGAŃ MODELU
    # ---------------------------------------------------------
    # 1. Tworzymy DataFrame z otrzymanego JSONa
    df_raw = pd.DataFrame([data])

    # 2. Pobieramy listę kolumn, na których model był faktycznie trenowany
    expected_features = list(rf_model.feature_names_in_)

    # 3. Tworzymy "pusty" DataFrame z dokładnie takimi samymi kolumnami, wypełniony zerami (lub medianami)
    df_model_input = pd.DataFrame(0, index=[0], columns=expected_features)

    # 4. Podmieniamy zera na wartości z JSON-a (tylko dla tych kolumn, które pasują)
    for col in expected_features:
        if col in df_raw.columns:
            df_model_input[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)

    # W tym momencie df_model_input ma IDEALNĄ strukturę dla modelu
    # ---------------------------------------------------------

    # =========================================================
    # LOGIKA TESTU A/B (Przezroczysty router modeli)
    # =========================================================
    selected_model = random.choice(["Model_A_IsolationForest", "Model_B_RandomForest"])

    try:
        # Predykcja używająca poprawnie sformatowanego df_model_input
        if selected_model == "Model_A_IsolationForest":
            pred_raw = iso_model.predict(df_model_input)[0]
            prediction = 1 if pred_raw == -1 else 0
        else:
            prediction = int(rf_model.predict(df_model_input)[0])

        # Zapisanie logu
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, offer_id, selected_model, prediction])

        return {
            "offer_id": offer_id,
            "anomaly_detected": bool(prediction),
            "status": "success",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
