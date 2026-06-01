# Nocarz - Detekcja Zawyżonych Cen (Anomalii) w Ofertach Noclegowych

Projekt realizowany w ramach przedmiotu IUM (Inżynieria Uczenia Maszynowego).
Celem systemu jest automatyczne wykrywanie ofert noclegowych, w których host celowo zawyża cenę (tzw. "odstraszacz"), aby zablokować kalendarz rezerwacji, unikając oznaczania terminu jako niedostępny.

Projekt opiera się na analizie zachowań użytkowników (sesje: view-to-book) oraz cechach fizycznych ofert. W ramach rozwiązania dostarczono pełen potok analityczny od eksploracji danych, przez modelowanie, aż po wdrożenie produkcyjne w architekturze REST API wraz z obsługą testów A/B.

---

## Struktura Projektu i Pliki

Rozwiązanie zostało podzielone na logiczne etapy zgodnie z wytycznymi projektu:

* **`IUM – Machine Learning Canvas (v1.0).pdf`** Dokumentacja biznesowo-analityczna projektu. Definiuje problem, kryteria sukcesu (Swoistość >= 90%, Czułość >= 30%) oraz założenia systemu.

### Notatniki Jupyter (Proces Analityczny)
Kolejność uruchamiania i czytania notatników jest odzwierciedleniem potoku danych (Data Pipeline):

1. **`01_Data_Analysis_Report.ipynb`**
   Raport z Eksploracyjnej Analizy Danych (EDA). Weryfikacja jakości zbiorów, spójności relacyjnej, analiza braków danych oraz kluczowa dla projektu analiza konwersji względem ceny (skala logarytmiczna), która udowadnia hipotezę biznesową.
2. **`02_Data_Transformation_and_Feature_Engineering.ipynb`**
   Inżynieria cech. Skrypt agregujący opinie, kodujący udogodnienia i łączący tabele. Jego wynikiem jest gotowy zbiór `production_dataset.csv`.
3. **`03_Modeling_and_Evaluation.ipynb`**
   Właściwe modelowanie i selekcja ostatecznego modelu. Zestawienie nienadzorowanego algorytmu bazowego (*Isolation Forest*) z wysoce zoptymalizowanym modelem nadzorowanym (*Random Forest*). Skrypt optymalizuje autorską metrykę Swoistości i eksportuje gotowe modele.
4. **`04_AB_Test_Evaluation.ipynb`**
   Skrypt do oceny statystycznej (Test Chi-kwadrat) przeprowadzonych testów A/B na podstawie logów z mikroserwisu.

### Kod Produkcyjny i Artefakty
* **`app.py`** - Mikroserwis napisany we frameworku *FastAPI*. Serwuje predykcje przez endpoint REST, dynamicznie mapuje wejścia i ukrycie obsługuje podział ruchu (test A/B).
* **`generated/`** - Katalog zawierający wyuczone wtyczki modelu w formacie `.joblib` (`isolation_forest_baseline.joblib` oraz `random_forest_production.joblib`).
* **`ab_test_logs.csv`** - Płaski plik bazodanowy generowany i nadpisywany przez `app.py`, służący jako rejestr zdarzeń w teście A/B.
* **`dowod_dzialania_api.png`** - Zrzut ekranu poświadczający pomyślne wykonanie predykcji za pomocą klienta `curl`.

---

## Jak korzystać z mikroserwisu (API)?

Aplikacja wymaga środowiska z zainstalowanymi bibliotekami: `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `joblib`. 
*(Można je doinstalować komendą: `pip install fastapi uvicorn pandas scikit-learn joblib`)*

### 1. Uruchomienie Serwera
Aby wystartować serwer API, otwórz terminal w głównym katalogu projektu i wpisz:
```bash
uvicorn app:app --port 8080
```
Serwer uruchomi się lokalnie i załaduje wyuczone modele ML. Pojawi się informacja `Application startup complete`.

### 2. Wykonywanie zapytań predykcyjnych (CURL)
Mikroserwis wystawia główny endpoint `/predict_price_anomaly`. Przyjmuje on dane w formacie JSON. 
Zaletą implementacji jest jej **odporność na braki w danych** - klient nie musi przesyłać wszystkich kolumn, API w locie uzupełni luki odpowiednimi wartościami neutralnymi (zgodnie z listą cech modelu).

Aby przetestować działanie, otwórz nowe okno terminala i wyślij próbne zapytanie za pomocą `curl`:

```bash
curl -X POST -H "Content-Type:application/json" -d '{"id": "12345", "price": 1500, "accommodates": 2, "bedrooms": 1, "beds": 1, "number_of_reviews": 5, "review_scores_rating": 4.5}' http://localhost:8080/predict_price_anomaly
```

**Spodziewana odpowiedź serwera (JSON):**
```json
{
  "offer_id": "12345",
  "anomaly_detected": true,
  "status": "success"
}
```
*Gdzie `anomaly_detected: true` oznacza, że dany model zakwalifikował ofertę jako posiadającą celowo zawyżoną cenę.*

### 3. Transparentny Test A/B
Wysyłając zapytanie JSON, nie wiesz, który z modeli przygotował odpowiedź (pełna przezroczystość dla usług zewnętrznych). 
Pod spodem mikroserwis z każdym zapytaniem **losuje w proporcjach 50/50** pomiędzy:
* Modelem A (Isolation Forest - Baseline)
* Modelem B (Random Forest - Ostateczny, wyoptymalizowany)

Z każdym zapytaniem `curl`, aplikacja zapisuje fakt obsługi do pliku `ab_test_logs.csv` zawierającego stempel czasowy, id oferty, nazwę użytego modelu oraz decyzję (0 - OK, 1 - Anomalia). Aby przetestować spójność, po kilku strzałach w API zajrzyj do w/w pliku .csv, a następnie uruchom notatnik `04_AB_Test_Evaluation.ipynb`, który podda te wyniki ocenie statystycznej.