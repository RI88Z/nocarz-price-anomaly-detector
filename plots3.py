import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# Wczytanie danych
# ==========================================

sessions = pd.read_csv("data/sessions.csv")
listings = pd.read_csv("data/listings.csv")

output_dir = Path("generated/rozklad_cen")
output_dir.mkdir(exist_ok=True)

# ==========================================
# Unikalne wartości action
# ==========================================

# print("Unikalne wartości w kolumnie action:")
# print(sessions["action"].unique())

# ==========================================
# Czyszczenie cen
# ==========================================

listings["price"] = listings["price"].astype(str).str.replace(r"[\$,]", "", regex=True)

listings["price"] = pd.to_numeric(listings["price"], errors="coerce")

# ==========================================
# Mapa listing_id -> cena
# ==========================================

listing_price_map = listings.set_index("id")["price"]

# ==========================================
# Funkcja pobierająca ceny
# ==========================================


def get_prices_for_action(action_name):
    df = sessions[sessions["action"] == action_name].copy()

    # Cena z sessions.csv
    if "price" in df.columns:
        df["price"] = df["price"].astype(str).str.replace(r"[\$,]", "", regex=True)

        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Cena z listings.csv
    df["listing_price"] = df["listing_id"].map(listing_price_map)

    # Finalna cena
    df["final_price"] = df["price"].fillna(df["listing_price"])

    prices = df["final_price"].dropna()

    # tylko dodatnie
    prices = prices[prices > 0]

    # usunięcie ekstremalnych outlierów
    prices = prices[prices < 5000]

    return prices


# ==========================================
# Dane
# ==========================================

view_prices = get_prices_for_action("view_listing")
book_prices = get_prices_for_action("book_listing")
cancel_prices = get_prices_for_action("cancel_booking")

# ==========================================
# Wspólne biny
# ==========================================

all_prices = pd.concat([view_prices, book_prices, cancel_prices])

min_price = all_prices.min()
max_price = all_prices.max()

# zwykłe biny
linear_bins = np.linspace(min_price, max_price, 60)

# logarytmiczne biny
log_bins = np.logspace(np.log10(min_price), np.log10(max_price), 60)

# ==========================================
# WYKRES 1:
# Porównanie rozkładów
# ==========================================


# ==========================================
# WYKRES porównawczy
# ==========================================


def create_combined_plot(log_scale=False):

    current_bins = log_bins if log_scale else linear_bins

    plt.figure(figsize=(14, 8))

    plt.hist(
        view_prices,
        bins=current_bins,
        alpha=0.65,
        label="view_listing",
        color="blue",
        edgecolor="black",
        density=True,
    )

    plt.hist(
        book_prices,
        bins=current_bins,
        alpha=0.65,
        label="book_listing",
        color="orange",
        edgecolor="black",
        density=True,
    )

    # plt.hist(
    #     cancel_prices,
    #     bins=current_bins,
    #     alpha=0.45,
    #     label="cancel_booking",
    #     color="salmon",
    #     edgecolor="black",
    #     density=True,
    # )

    if log_scale:
        plt.xscale("log")
        filename = "combined_distribution_logx.png"
        plt.title("Porównanie rozkładów cen (skala logarytmiczna)")
    else:
        filename = "combined_distribution.png"
        plt.title("Porównanie rozkładów cen")

    plt.xlabel("Cena")
    plt.ylabel("Gęstość rozkładu")

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Zapisano: {filename}")


# ==========================================
# WYKRES 2:
# % viewed -> booked
# ==========================================


# ==========================================
# WYKRES konwersji
# ==========================================


def create_conversion_plot(log_scale=False):

    current_bins = log_bins if log_scale else linear_bins

    # histogramy liczności
    view_counts, edges = np.histogram(view_prices, bins=current_bins)

    book_counts, _ = np.histogram(book_prices, bins=current_bins)

    # konwersja %
    conversion = (
        np.divide(
            book_counts,
            view_counts,
            out=np.zeros_like(book_counts, dtype=float),
            where=view_counts != 0,
        )
        * 100
    )

    # środki binów
    centers = (edges[:-1] + edges[1:]) / 2

    plt.figure(figsize=(14, 8))

    plt.bar(
        centers,
        conversion,
        width=np.diff(edges),
        color="mediumpurple",
        edgecolor="black",
        alpha=0.8,
        align="center",
    )

    if log_scale:
        plt.xscale("log")
        filename = "view_to_book_conversion_logx.png"
        plt.title(
            "Odsetek zarezerwowanych ofert wśród oglądanych (skala logarytmiczna)"
        )
    else:
        filename = "view_to_book_conversion.png"
        plt.title("Odsetek zarezerwowanych ofert wśród oglądanych")

    plt.xlabel("Cena")
    plt.ylabel("Odsetek zarezerwowanych ofert wśród oglądanych (%)")

    plt.tight_layout()

    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Zapisano: {filename}")


# ==========================================
# Generowanie wykresów
# ==========================================

create_combined_plot(log_scale=False)
create_combined_plot(log_scale=True)

create_conversion_plot(log_scale=False)
create_conversion_plot(log_scale=True)

print("\nGotowe.")
