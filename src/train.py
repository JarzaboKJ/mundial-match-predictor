import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier

from preprocessing import (
    load_and_clean_matches,
    build_team_stats,
    build_h2h_stats,
    build_feature_matrix,
)

FEATURE_COLS = [
    "rank_diff",
    "points_diff",
    "home_win_rate",
    "away_win_rate",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "stage",
    "home_appearances",
    "away_appearances",
    "home_champion_count",
    "away_champion_count",
    "home_final_appearances",
    "away_final_appearances",
]

CLASS_NAMES = ["home_win", "draw", "away_win"]


def compute_metrics(name, y_true, y_pred, y_proba):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_proba)
    roc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    brier = {}
    for i, cls in enumerate(CLASS_NAMES):
        brier[cls] = brier_score_loss(y_bin[:, i], y_proba[:, i])

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    draw_recall = report["draw"]["recall"]

    return {
        "name": name,
        "accuracy": acc,
        "log_loss": ll,
        "roc_auc_macro": roc,
        "brier_home_win": brier["home_win"],
        "brier_draw": brier["draw"],
        "brier_away_win": brier["away_win"],
        "draw_recall": draw_recall,
    }


def print_metrics_table(rows):
    metrics = [
        ("Accuracy", "accuracy", False),
        ("Log Loss", "log_loss", True),
        ("ROC AUC (macro)", "roc_auc_macro", False),
        ("Brier — home_win", "brier_home_win", True),
        ("Brier — draw", "brier_draw", True),
        ("Brier — away_win", "brier_away_win", True),
        ("Draw Recall", "draw_recall", False),
    ]

    col_w = max(len(r["name"]) for r in rows) + 2
    header = f"{'Metric':<25s}" + "".join(f"{r['name']:>{col_w}s}" for r in rows)
    print(header)
    print("-" * len(header))
    for label, key, lower_better in metrics:
        vals = [r[key] for r in rows]
        best = min(vals) if lower_better else max(vals)
        parts = []
        for v in vals:
            mark = " *" if v == best and len(vals) > 1 else "  "
            parts.append(f"{v:>{col_w - 2}.4f}{mark}")
        print(f"{label:<25s}" + "".join(parts))
    print("(* = better)\n")


def main():
    print("Loading and cleaning matches...")
    df = load_and_clean_matches()
    print(f"  {len(df)} matches loaded")

    print("Building team stats...")
    team_stats = build_team_stats(df)

    print("Building head-to-head stats...")
    h2h_stats = build_h2h_stats(df)

    print("Loading ranking data (2022 as historical proxy)...")
    ranking_df = pd.read_csv("data/raw/fifa_ranking_2022-10-06.csv")

    print("Building feature matrix...")
    features_df = build_feature_matrix(df, team_stats, h2h_stats, ranking_df)
    features_df.to_csv("data/processed/features.csv", index=False)
    print(f"  Feature matrix saved: {len(features_df)} rows, {len(FEATURE_COLS)} features")

    X = features_df[FEATURE_COLS].values
    y = features_df["target"].values
    sample_weights = np.where(features_df["Year"] >= 2010, 2.0, 1.0)

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )

    # ── Baseline training (before tuning) ────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE TRAINING (default hyperparameters)")
    print("=" * 60)

    xgb_base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb_base.fit(X_train, y_train, sample_weight=w_train)
    y_pred_xgb_base = xgb_base.predict(X_test)
    y_proba_xgb_base = xgb_base.predict_proba(X_test)

    lr_base = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    )
    lr_base.fit(X_train, y_train, logisticregression__sample_weight=w_train)
    y_pred_lr_base = lr_base.predict(X_test)
    y_proba_lr_base = lr_base.predict_proba(X_test)

    m_xgb_base = compute_metrics("XGB baseline", y_test, y_pred_xgb_base, y_proba_xgb_base)
    m_lr_base = compute_metrics("LR baseline", y_test, y_pred_lr_base, y_proba_lr_base)

    print("\nBaseline comparison:")
    print_metrics_table([m_xgb_base, m_lr_base])

    print("XGBoost Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb_base))
    print("\nLR Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr_base))

    print("\nTop 10 Feature Importances (XGBoost baseline):")
    importances = xgb_base.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    for i in idx:
        print(f"  {FEATURE_COLS[i]:30s} {importances[i]:.4f}")

    # ── Hyperparameter tuning ────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING — XGBoost (RandomizedSearchCV)")
    print("=" * 60)

    xgb_param_space = {
        "max_depth": [3, 4, 5, 6, 7],
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.15],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    }

    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            eval_metric="mlogloss",
        ),
        param_distributions=xgb_param_space,
        n_iter=30,
        cv=cv,
        scoring="neg_log_loss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train, sample_weight=w_train)
    print(f"Best params: {xgb_search.best_params_}")
    print(f"Best CV log_loss: {-xgb_search.best_score_:.4f}")

    xgb_tuned = xgb_search.best_estimator_
    y_pred_xgb_tuned = xgb_tuned.predict(X_test)
    y_proba_xgb_tuned = xgb_tuned.predict_proba(X_test)

    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING — Logistic Regression (GridSearchCV)")
    print("=" * 60)

    lr_pipe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    )
    lr_param_grid = {
        "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    }
    lr_search = GridSearchCV(
        lr_pipe,
        param_grid=lr_param_grid,
        cv=cv,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    lr_search.fit(X_train, y_train, logisticregression__sample_weight=w_train)
    print(f"Best params: {lr_search.best_params_}")
    print(f"Best CV log_loss: {-lr_search.best_score_:.4f}")

    lr_tuned = lr_search.best_estimator_
    y_pred_lr_tuned = lr_tuned.predict(X_test)
    y_proba_lr_tuned = lr_tuned.predict_proba(X_test)

    # ── Final comparison ─────────────────────────────────────────
    m_xgb_tuned = compute_metrics("XGB tuned", y_test, y_pred_xgb_tuned, y_proba_xgb_tuned)
    m_lr_tuned = compute_metrics("LR tuned", y_test, y_pred_lr_tuned, y_proba_lr_tuned)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON — all models")
    print("=" * 60 + "\n")
    print_metrics_table([m_xgb_base, m_xgb_tuned, m_lr_base, m_lr_tuned])

    print("XGBoost tuned — Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_xgb_tuned))
    print("\nXGBoost tuned — Classification Report:")
    print(classification_report(y_test, y_pred_xgb_tuned, target_names=CLASS_NAMES))

    print("LR tuned — Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr_tuned))
    print("\nLR tuned — Classification Report:")
    print(classification_report(y_test, y_pred_lr_tuned, target_names=CLASS_NAMES))

    print("Top 10 Feature Importances (XGBoost tuned):")
    importances = xgb_tuned.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    for i in idx:
        print(f"  {FEATURE_COLS[i]:30s} {importances[i]:.4f}")

    # ── Save best models ─────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_tuned, "models/model.pkl")
    joblib.dump(lr_tuned, "models/baseline_model.pkl")
    print("\nModels saved:")
    print("  models/model.pkl (XGBoost tuned)")
    print("  models/baseline_model.pkl (LR tuned)")


if __name__ == "__main__":
    main()
