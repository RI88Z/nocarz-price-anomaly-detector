# ============================================================
# PRODUCTION-GRADE NESTED CV MODEL SELECTION
# PRIMARY METRIC: SPECIFICITY (HIGH FP COST)
# CONSTRAINT: RECALL >= 30%
# ============================================================

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from scipy.stats import randint, uniform

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    classification_report,
    make_scorer,
)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 5
N_ITER_SEARCH = 100
MIN_RECALL = 0.30
TARGET = "overpriced"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/production_dataset.csv")

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Dataset shape: {df.shape}")
print("\nTarget distribution:")
print(y.value_counts(normalize=True))


# ============================================================
# CLASS IMBALANCE
# ============================================================

n_pos = (y == 1).sum()
n_neg = (y == 0).sum()
scale_pos_weight = n_neg / n_pos

print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")


# ============================================================
# CV
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

inner_cv = StratifiedKFold(
    n_splits=INNER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# SPECIFICITY METRIC
# ============================================================


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)


specificity_scorer = make_scorer(specificity_score)


# ============================================================
# MODELS
# ============================================================

models = {
    "RandomForest": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "n_estimators": randint(200, 1200),
            "max_depth": [None, 5, 8, 10, 15, 20, 30],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        },
    ),
    "ExtraTrees": (
        ExtraTreesClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "n_estimators": randint(200, 1200),
            "max_depth": [None, 5, 8, 10, 15, 20, 30],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        },
    ),
    "XGBoost": (
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        {
            "n_estimators": randint(200, 1000),
            "max_depth": randint(3, 12),
            "learning_rate": uniform(0.01, 0.29),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "gamma": uniform(0, 5),
            "min_child_weight": randint(1, 10),
        },
    ),
}


# ============================================================
# NESTED CV
# ============================================================

results = []

best_model_name = None
best_specificity = -np.inf
best_oof_predictions = None

print("\n" + "=" * 80)
print("STARTING NESTED CROSS VALIDATION")
print("=" * 80)


for model_name, (model, param_space) in models.items():

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    outer_specificity = []
    outer_recall = []
    outer_precision = []
    outer_f1 = []
    outer_accuracy = []

    oof_preds = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=N_ITER_SEARCH,
            scoring=specificity_scorer,
            cv=inner_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        preds = best_model.predict(X_test)

        oof_preds[test_idx] = preds

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        outer_specificity.append(spec)
        outer_recall.append(recall_score(y_test, preds))
        outer_precision.append(precision_score(y_test, preds))
        outer_f1.append(f1_score(y_test, preds))
        outer_accuracy.append(accuracy_score(y_test, preds))

        print(f"Fold {fold} | Specificity={spec:.4f}")

    mean_specificity = np.mean(outer_specificity)
    mean_recall = np.mean(outer_recall)
    mean_precision = np.mean(outer_precision)
    mean_f1 = np.mean(outer_f1)
    mean_accuracy = np.mean(outer_accuracy)

    results.append(
        {
            "model": model_name,
            "specificity_mean": mean_specificity,
            "recall_mean": mean_recall,
            "precision_mean": mean_precision,
            "f1_mean": mean_f1,
            "accuracy_mean": mean_accuracy,
        }
    )

    print("\nNested CV Results:")
    print(f"Specificity : {mean_specificity:.4f}")
    print(f"Recall      : {mean_recall:.4f}")
    print(f"Precision   : {mean_precision:.4f}")
    print(f"F1          : {mean_f1:.4f}")
    print(f"Accuracy    : {mean_accuracy:.4f}")

    # ========================================================
    # SELECTION RULE
    # ========================================================

    if mean_recall >= MIN_RECALL:
        if mean_specificity > best_specificity:
            best_specificity = mean_specificity
            best_model_name = model_name
            best_oof_predictions = oof_preds


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results).sort_values(
    by="specificity_mean",
    ascending=False,
)

print("\n" + "=" * 80)
print("MODEL RANKING (BY SPECIFICITY)")
print("=" * 80)
print(results_df)


# ============================================================
# BEST MODEL CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y, best_oof_predictions)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

print("\n" + "=" * 80)
print("CONFUSION MATRIX (TERMINAL)")
print("=" * 80)

print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print(f"\nSpecificity: {specificity:.4f}")


plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"OOF Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("\nClassification Report:\n")
print(classification_report(y, best_oof_predictions, zero_division=0))


# ============================================================
# FINAL TRAINING
# ============================================================

print("\nRetraining final model...")

best_model_final = models[best_model_name][0]
best_param_space = models[best_model_name][1]

final_search = RandomizedSearchCV(
    estimator=best_model_final,
    param_distributions=best_param_space,
    n_iter=N_ITER_SEARCH,
    scoring=specificity_scorer,
    cv=inner_cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)

final_search.fit(X, y)

final_model = final_search.best_estimator_

joblib.dump(final_model, f"generated/{best_model_name.lower()}_production.joblib")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL MODEL")
print("=" * 80)

print("Model:", best_model_name)
print("CV Specificity:", best_specificity)
print("Saved to:", f"generated/{best_model_name.lower()}_production.joblib")


results_df.to_csv(
    "generated/model_comparison_results.csv",
    index=False,
)

print("\nSaved results to CSV")
