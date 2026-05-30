import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =========================
# Wczytanie danych
# =========================
production_df = pd.read_csv("data/production_dataset.csv")


# =========================
# Funkcja do oceny datasetu
# =========================
def evaluate_dataset(df, dataset_name):
    X = df.drop(columns=["overpriced"])
    y = df["overpriced"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = []
    prec_scores = []
    rec_scores = []
    cms = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            # max_depth=6,
            # min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        acc_scores.append(accuracy_score(y_val, y_val_pred))
        prec_scores.append(precision_score(y_val, y_val_pred, zero_division=0))
        rec_scores.append(recall_score(y_val, y_val_pred, zero_division=0))
        cms.append(confusion_matrix(y_val, y_val_pred))

        print(
            f"{dataset_name} fold={fold} | "
            f"Accuracy={acc_scores[-1]:.3f} "
            f"Precision={prec_scores[-1]:.3f} "
            f"Recall={rec_scores[-1]:.3f}"
        )

    print("\n==============================")
    print(dataset_name)
    print("==============================")
    print(f"Accuracy : {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
    print(f"Precision: {np.mean(prec_scores):.3f} ± {np.std(prec_scores):.3f}")
    print(f"Recall   : {np.mean(rec_scores):.3f} ± {np.std(rec_scores):.3f}")

    avg_cm = np.mean(cms, axis=0)
    # print("\nAverage confusion matrix:\n", avg_cm)

    # wykres confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap="Blues")
    plt.title(f"{dataset_name} - Average CV Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return avg_cm


# =========================
# Ewaluacja modelu
# =========================
cm_prod = evaluate_dataset(production_df, "Production Dataset")

# =========================
# Trening finalnego modelu na całym zbiorze
# =========================

X = production_df.drop(columns=["overpriced"])
y = production_df["overpriced"]

final_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

final_model.fit(X, y)

# zapis modelu
joblib.dump(final_model, "generated/random_forest_production.joblib")

print("Model zapisany pomyślnie")
