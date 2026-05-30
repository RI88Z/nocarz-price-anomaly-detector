import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

files = ["calendar.csv", "listings.csv", "reviews.csv", "sessions.csv", "users.csv"]

output_dir = Path("generated/raport_brakow")
output_dir.mkdir(exist_ok=True)

report_lines = []
rows_missing_summary = {}

for file in files:
    print(f"Analiza: {file}")
    df = pd.read_csv("data/" + file)

    # ==============================
    # 1) % braków w kolumnach (5 wykresów)
    # ==============================
    missing_col_percent = df.isna().mean() * 100
    # missing_col_percent = missing_col_percent.sort_values(ascending=False)

    report_lines.append(f"\n===== {file} =====")
    report_lines.append("Procent braków w kolumnach:")
    report_lines.append(missing_col_percent.to_string())

    plt.figure()
    missing_col_percent.plot(kind="bar")
    # dynamiczna wysokość wykresu zależna od liczby kolumn
    fig_height = max(6, len(missing_col_percent) * 0.35)

    plt.figure(figsize=(fig_height, 10))

    missing_col_percent.sort_values().plot(kind="barh")

    plt.title(f"% braków w kolumnach - {file}")
    plt.xlabel("% brakujących danych")
    plt.ylabel("Kolumny")

    plt.tight_layout()
    plt.savefig(output_dir / f"{file}_missing_columns.png")
    plt.close()

    # ==============================
    # 2) % wierszy z brakami (do wykresu zbiorczego)
    # ==============================
    rows_with_missing = df.isna().any(axis=1).mean() * 100
    rows_missing_summary[file] = rows_with_missing

    report_lines.append(f"\n% wierszy z brakami: {rows_with_missing:.2f}%")

# ==============================
# 3) Wspólny wykres dla wierszy (1 wykres)
# ==============================
plt.figure()
plt.bar(rows_missing_summary.keys(), rows_missing_summary.values())
plt.title("Procent wierszy zawierających braki w danych")
plt.ylabel("% wierszy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(output_dir / "ALL_files_missing_rows.png")
plt.close()

# ==============================
# 4) zapis raportu tekstowego
# ==============================
with open(output_dir / "raport.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\nGotowe! Powstało 6 wykresów + raport.txt w folderze 'generated/raport_brakow'")
