import csv
import os
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request

# 1. Inicjalizacja aplikacji FastAPI
app = FastAPI(
    title="Nocarz API",
    description="Mikroserwis do detekcji anomalii cenowych po transformacjach",
)

print("Wczytywanie modeli i artefaktów...")
# Upewnij się, że nazwa pliku modelu zgadza się z tym z notatnika 03 (np. best_model_production.joblib lub extratrees_production.joblib)
try:
    rf_model = joblib.load("generated/best_model_production.joblib")
    iso_model = joblib.load("generated/isolation_forest_baseline.joblib")

    # Wczytanie artefaktów transformacji
    artifacts = joblib.load("generated/artifacts.joblib")
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    amenity_freq = artifacts["amenity_freq"]
    log_cols = artifacts["log_cols"]
    bin_cols = artifacts["bin_cols"]
    other_cols = artifacts["other_cols"]
except FileNotFoundError as e:
    print(f"Błąd krytyczny: Nie znaleziono pliku! {e}")

LOG_FILE = "ab_test_logs.csv"

# 2. Inicjalizacja pliku z logami testu A/B
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["timestamp", "offer_id", "model_assigned", "prediction_result"]
        )


# 3. Funkcja pomocnicza do zliczania punktacji udogodnień
def process_amenities(amenities, amenity_freq_dict):
    if not isinstance(amenities, list):
        return 0, 0.0

    amenities_count = len(amenities)
    premium_score = 0
    for a in amenities:
        freq = amenity_freq_dict.get(a, 0.001)
        premium_score += 1 / freq
    premium_score = premium_score / amenities_count if amenities_count > 0 else 0
    return amenities_count, premium_score


# 4. Główny endpoint przyjmujący zapytania (metoda POST)
@app.post("/predict_price_anomaly")
async def predict_price(request: Request):
    data = await request.json()
    offer_id = data.get("id", "unknown_id")

    try:
        # =========================================================
        # ODTWORZENIE TRANSFORMACJI (Data Pipeline)
        # =========================================================

        # 1. Przetworzenie udogodnień
        amenities = data.get("amenities", [])
        amenities_count, amenities_premium_score = process_amenities(
            amenities, amenity_freq
        )

        # 2. Mapowanie Room Type
        room_type = data.get("room_type", "Entire home/apt")
        room_type_mapping = {"Entire home/apt": 0, "Private room": 1}
        mapped_room_type = room_type_mapping.get(room_type, 2)

        # 3. Zbudowanie podstawowego słownika cech z zapytania
        input_dict = {
            "price": data.get("price", 0.0),
            "accommodates": data.get("accommodates", 0),
            "bedrooms": data.get("bedrooms", 0.0),
            "bathrooms": data.get("bathrooms", 0.0),
            "beds": data.get("beds", 0.0),
            "room_type": mapped_room_type,
            "latitude": data.get("latitude", 0.0),
            "longitude": data.get("longitude", 0.0),
            "host_response_rate": data.get("host_response_rate", 0.0),
            "host_acceptance_rate": data.get("host_acceptance_rate", 0.0),
            "host_total_listings_count": data.get("host_total_listings_count", 0),
            "amenities_count": amenities_count,
            "amenities_premium_score": amenities_premium_score,
        }

        # Doklejenie ew. pozostałych kluczy, jeśli zostały wysłane z zewnątrz
        for key, val in data.items():
            if key not in input_dict and key in feature_names:
                input_dict[key] = val

        input_df = pd.DataFrame([input_dict])

        # 4. FULL CANVAS: Wypełnienie ramki zerami dla brakujących cech
        full_df = pd.DataFrame(0, columns=feature_names, index=input_df.index)
        common_cols = input_df.columns.intersection(full_df.columns)
        full_df[common_cols] = (
            input_df[common_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        )

        # 5. TRANSFORMACJE LOGARYTMICZNE
        # Zabezpieczenie przed liczbami ujemnymi przy log1p
        for col in log_cols:
            if col in full_df.columns:
                full_df[col] = np.log1p(full_df[col].clip(lower=0))

        # 6. SKALOWANIE DANYCH
        scaler_cols = scaler.feature_names_in_
        valid_scaler_cols = [c for c in scaler_cols if c in full_df.columns]
        if len(valid_scaler_cols) == len(scaler_cols):
            full_df[scaler_cols] = scaler.transform(full_df[scaler_cols])

        # =========================================================
        # LOGIKA TESTU A/B
        # =========================================================
        # Pamiętajmy, by wyciąć z full_df dokładnie te kolumny, których wymaga dany model
        selected_model = random.choice(
            ["Model_A_IsolationForest", "Model_B_ExtraTrees"]
        )

        if selected_model == "Model_A_IsolationForest":
            expected_features = list(iso_model.feature_names_in_)
            df_model_input = full_df[expected_features]
            pred_raw = iso_model.predict(df_model_input)[0]
            prediction = 1 if pred_raw == -1 else 0
        else:
            expected_features = list(rf_model.feature_names_in_)
            df_model_input = full_df[expected_features]
            prediction = int(rf_model.predict(df_model_input)[0])

        # =========================================================
        # REJESTRACJA W LOGACH
        # =========================================================
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
        import traceback

        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
