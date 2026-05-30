import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

output_dir = Path("generated/raport_jakosci")
output_dir.mkdir(exist_ok=True)

calendar = pd.read_csv("data/calendar.csv")
reviews = pd.read_csv("data/reviews.csv")
sessions = pd.read_csv("data/sessions.csv")
listings = pd.read_csv("data/listings.csv")

report = []


def pokrycie():
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    sessions["timestamp"] = pd.to_datetime(sessions["timestamp"], errors="coerce")

    cal_month = calendar.set_index("date").resample("ME").size()
    rev_month = reviews.set_index("date").resample("ME").size()
    ses_month = sessions.set_index("timestamp").resample("ME").size()

    plt.figure(figsize=(12, 6))
    plt.plot(cal_month, label="calendar")
    plt.plot(rev_month, label="reviews")
    plt.plot(ses_month, label="sessions")
    plt.title("Pokrycie danych w czasie (liczba rekordów miesięcznie)")
    plt.ylabel("Liczba rekordów")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_coverage.png")
    plt.close()

    report.append("=== Zakres dat ===")
    report.append(f"calendar: {calendar['date'].min()} -> {calendar['date'].max()}")
    report.append(f"reviews: {reviews['date'].min()} -> {reviews['date'].max()}")
    report.append(
        f"sessions: {sessions['timestamp'].min()} -> {sessions['timestamp'].max()}"
    )


def rozklad_log():
    calendar["price_clean"] = (
        calendar["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    prices = calendar["price_clean"].dropna()

    # usuwamy zera i wartości ujemne (log nie działa dla <=0)
    prices = prices[prices > 0]

    # zakres logarytmiczny
    min_price = prices.min()
    max_price = prices.max()

    # logarytmiczne binsy (np. 50 koszyków)
    bins = np.logspace(np.log10(min_price), np.log10(max_price), 50)

    plt.figure(figsize=(10, 6))
    plt.hist(prices, bins=bins)

    plt.xscale("log")
    plt.title("Rozkład cen ofert")
    plt.xlabel("Cena (USD) za ofertę")
    plt.ylabel("Liczba rekordów")

    plt.tight_layout()
    plt.savefig(output_dir / "price_distribution_log.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.boxplot(calendar["price_clean"].dropna(), vert=True)
    plt.ylabel("Cena (USD) za ofertę")
    plt.title("Boxplot cen ofert")
    plt.tight_layout()
    plt.savefig(output_dir / "price_boxplot.png")
    plt.close()

    report.append("\n=== Statystyki cen ===")
    report.append(calendar["price_clean"].describe().to_string())


def rozklad():
    calendar["price_clean"] = (
        calendar["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    prices = calendar["price_clean"].dropna()

    # usuwamy zera i wartości ujemne (log nie działa dla <=0)
    prices = prices[prices > 0]

    # zakres logarytmiczny
    # min_price = 10000
    min_price = prices.min()
    max_price = prices.max()
    # max_price = 30000

    # binsy (np. 50 koszyków)
    bins = np.linspace(min_price, max_price, 100)

    plt.figure(figsize=(10, 6))
    plt.hist(prices, bins=bins)

    # plt.xscale("log")
    plt.title("Rozkład cen ofert")
    plt.xlabel("Cena (USD) za ofertę")
    plt.ylabel("Liczba rekordów")

    plt.tight_layout()
    plt.savefig(output_dir / "price_distribution.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.boxplot(calendar["price_clean"].dropna(), vert=True)
    plt.ylabel("Cena (USD) za ofertę")
    plt.title("Boxplot cen ofert")
    plt.tight_layout()
    plt.savefig(output_dir / "price_boxplot.png")
    plt.close()

    report.append("\n=== Statystyki cen ===")
    report.append(calendar["price_clean"].describe().to_string())


def spojnosc():
    calendar_ids = set(calendar["listing_id"].dropna())
    listings_ids = set(listings["id"].dropna())
    reviews_ids = set(reviews["listing_id"].dropna())
    sessions_ids = set(sessions["listing_id"].dropna())

    def missing_percent(source_ids, target_ids):
        missing = source_ids - target_ids
        return len(missing) / len(source_ids) * 100

    calendar_missing = missing_percent(calendar_ids, listings_ids)
    reviews_missing = missing_percent(reviews_ids, listings_ids)
    sessions_missing = missing_percent(sessions_ids, listings_ids)

    plt.figure(figsize=(8, 6))
    labels = ["calendar.listing_id", "reviews_listing_id", "sessions.listing_id"]
    values = [calendar_missing, reviews_missing, sessions_missing]

    plt.bar(labels, values)
    plt.ylabel("Procent rekordów bez dopasowania ID do listings.id")
    plt.title("Spójność relacji między tabelami")
    plt.tight_layout()
    plt.savefig(output_dir / "join_consistency.png")
    plt.close()

    report.append("\n=== Spójność ID ===")
    report.append(f"calendar bez listings: {calendar_missing:.2f}%")
    report.append(f"reviews bez listings: {reviews_missing:.2f}%")
    report.append(f"sessions bez listings: {sessions_missing:.2f}%")


if __name__ == "__main__":
    pokrycie()
    rozklad()
    rozklad_log()
    spojnosc()
    with open(output_dir / "raport_jakosci.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("Gotowe — wykresy i raport zapisane w folderze generated/raport_jakosci")
