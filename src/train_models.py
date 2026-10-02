"""
train_models.py
---------------
Trains and evaluates three binary classifiers (AVNRT = 0, AVRT = 1) on the
feature matrix produced by extract_features.py:

  * Logistic Regression  (median impute -> StandardScaler -> LR, L1/L2 via saga)
  * Random Forest        (median impute -> RF, class_weight='balanced')
  * XGBoost              (native NaN handling)

Protocol (small-n clinical dataset):
  * Stratified 80/20 train/test split (random_state=42)
  * 5-fold StratifiedKFold GridSearchCV for hyper-parameters (train split only)
  * Leave-One-Out CV on the full dataset as the primary metric
  * Out-of-fold XGBoost audit table listing every misclassified recording

Usage:
    python src/train_models.py --features features.xlsx --out-dir results

Author: Anurag Chatterjee
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, LeaveOneOut, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
SEED = 42

# -----------------------------------------------------------------------------
# Final feature set (19) - chosen from Mann-Whitney U significance testing,
# effect size and missing-value rate across the candidate features.
# -----------------------------------------------------------------------------
CH3_FEATURES = [
    # Tier 1 - strongest separators (low NaN, significant p-value)
    "pNN50_pct", "HR_std_bpm", "beat_skewness_mean", "beat_kurtosis_mean",
    "QRS_duration_ms_mean", "P_wave_presence_ratio", "ST_slope_mean", "R_peak_count",
    # Tier 2 - strong effect, ~30% NaN at 200+ BPM -> imputed
    "RP_interval_mean_ms", "RP_PR_ratio",
    # Tier 3 - moderate, statistically significant
    "SD2_ms", "SD1_ms", "SD1_SD2_ratio", "RR_std_ms", "RMSSD_ms", "RR_mean_ms",
    "P_wave_absent_flag",
]
CH4_FEATURES = ["SD1_SD2_ratio", "P_wave_absent_flag"]  # only ch4 features that were significant
FEATURES = [f"ch3_{f}" for f in CH3_FEATURES] + [f"ch4_{f}" for f in CH4_FEATURES]
CLASS_NAMES = ["AVNRT (0)", "AVRT (1)"]


def report(name, y_true, y_pred, y_prob=None):
    print(f"\n{'=' * 50}\n{name} - TEST SET\n{'=' * 50}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred):.4f}")
    if y_prob is not None:
        print(f"AUC-ROC  : {roc_auc_score(y_true, y_prob):.4f}")


def save_cm(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASS_NAMES).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def logistic_regression(X, y, Xtr, Xte, ytr, yte, cv, out):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Xtr_s, Xte_s = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))

    grid = GridSearchCV(LogisticRegression(max_iter=2000, random_state=SEED),
                        {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"], "solver": ["saga"]},
                        cv=cv, scoring="accuracy", n_jobs=-1).fit(Xtr_s, ytr)
    print(f"\n[LR] best params {grid.best_params_} | 5-fold CV acc {grid.best_score_:.4f}")
    model = grid.best_estimator_
    report("LOGISTIC REGRESSION", yte, model.predict(Xte_s), model.predict_proba(Xte_s)[:, 1])
    save_cm(yte, model.predict(Xte_s), "Logistic Regression - Confusion Matrix",
            os.path.join(out, "LR_confusion_matrix.png"))

    coef = pd.DataFrame({"feature": FEATURES, "coefficient": model.coef_[0]})
    coef = coef.reindex(coef.coefficient.abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(coef.feature, coef.coefficient,
            color=["tomato" if c > 0 else "steelblue" for c in coef.coefficient])
    ax.axvline(0, color="black", linewidth=0.8); ax.invert_yaxis()
    ax.set_xlabel("Coefficient (positive = predicts AVRT)")
    ax.set_title("Logistic Regression - Feature Coefficients")
    plt.tight_layout(); plt.savefig(os.path.join(out, "LR_feature_coefficients.png"), dpi=150); plt.close()

    loo_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("lr", LogisticRegression(**grid.best_params_, max_iter=2000, random_state=SEED))])
    loo = cross_val_score(loo_pipe, X, y, cv=LeaveOneOut(), scoring="accuracy")
    return grid.best_score_, loo


def random_forest(X, y, Xtr, Xte, ytr, yte, cv, out):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1))])
    grid = GridSearchCV(pipe, {"rf__n_estimators": [50, 100, 200], "rf__max_depth": [3, 5, None],
                               "rf__min_samples_split": [2, 5]},
                        cv=cv, scoring="accuracy", n_jobs=-1).fit(Xtr, ytr)
    print(f"\n[RF] best params {grid.best_params_} | 5-fold CV acc {grid.best_score_:.4f}")
    best = grid.best_estimator_
    report("RANDOM FOREST", yte, best.predict(Xte))
    save_cm(yte, best.predict(Xte), "Random Forest - Confusion Matrix", os.path.join(out, "RF_confusion_matrix.png"))

    imp = pd.Series(best.named_steps["rf"].feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nRandom Forest feature importances:\n" + imp.to_string())
    loo = cross_val_score(grid.best_estimator_, X, y, cv=LeaveOneOut(), scoring="accuracy")
    return grid.best_score_, loo


def make_xgb(**kw):
    return xgb.XGBClassifier(objective="binary:logistic", missing=np.nan, random_state=SEED,
                             eval_metric="logloss", n_jobs=-1, **kw)


def xgboost_model(X, y, Xtr, Xte, ytr, yte, cv, out):
    grid = GridSearchCV(make_xgb(), {"n_estimators": [50, 100], "max_depth": [2, 3, 4],
                                     "learning_rate": [0.01, 0.1, 0.2]},
                        cv=cv, scoring="accuracy", n_jobs=-1).fit(Xtr, ytr)
    print(f"\n[XGB] best params {grid.best_params_} | 5-fold CV acc {grid.best_score_:.4f}")
    best = grid.best_estimator_
    report("XGBOOST", yte, best.predict(Xte))
    save_cm(yte, best.predict(Xte), "XGBoost - Confusion Matrix", os.path.join(out, "XGB_confusion_matrix.png"))

    imp = pd.Series(best.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nXGBoost feature importances:\n" + imp.to_string())
    loo = cross_val_score(make_xgb(**grid.best_params_), X, y, cv=LeaveOneOut(), scoring="accuracy")
    return grid.best_score_, loo, grid.best_params_


def xgb_audit(X, y, ids, params, out):
    """Out-of-fold (LOO) predictions for every recording -> misclassification audit."""
    pred = np.zeros(len(X), dtype=int); prob = np.zeros(len(X))
    for tr, te in LeaveOneOut().split(X):
        m = make_xgb(**params).fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict(X.iloc[te]); prob[te] = m.predict_proba(X.iloc[te])[:, 1]
    lab = {0: "AVNRT", 1: "AVRT"}
    audit = pd.DataFrame({"recording_id": ids, "true": y.map(lab), "predicted": pd.Series(pred).map(lab),
                          "p_AVRT": prob})
    audit["correct"] = audit["true"] == audit["predicted"]
    print("\nMisclassified recordings (LOO, XGBoost):")
    print(audit[~audit.correct].to_string(index=False) or "None")
    audit.to_csv(os.path.join(out, "xgb_loo_audit.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="AVNRT vs AVRT classifiers")
    ap.add_argument("--features", default="features.xlsx")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = pd.read_excel(args.features)
    X, y = data[FEATURES], data["label_num"]
    ids = data["recording_id"] if "recording_id" in data else data.index.astype(str)
    print(f"{len(data)} recordings | {len(FEATURES)} features | class counts {y.value_counts().to_dict()}")
    print("NaN per feature:\n" + X.isna().sum()[X.isna().sum() > 0].to_string())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    summary = {}
    cv_lr, loo_lr = logistic_regression(X, y, Xtr, Xte, ytr, yte, cv, args.out_dir)
    summary["Logistic Regression"] = (cv_lr, loo_lr)
    cv_rf, loo_rf = random_forest(X, y, Xtr, Xte, ytr, yte, cv, args.out_dir)
    summary["Random Forest"] = (cv_rf, loo_rf)
    cv_xgb, loo_xgb, xgb_params = xgboost_model(X, y, Xtr, Xte, ytr, yte, cv, args.out_dir)
    summary["XGBoost"] = (cv_xgb, loo_xgb)

    print(f"\n{'=' * 64}\n{'Model':<22}{'LOO-CV acc':>12}{'5-fold CV acc':>16}{'Correct':>12}\n{'-' * 64}")
    for name, (cvs, loo) in summary.items():
        print(f"{name:<22}{loo.mean():>12.2%}{cvs:>16.2%}{f'{int(loo.sum())}/{len(loo)}':>12}")

    xgb_audit(X, y, ids, xgb_params, args.out_dir)


if __name__ == "__main__":
    main()
