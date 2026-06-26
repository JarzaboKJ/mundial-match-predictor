import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier

from preprocessing import (
    FEATURE_COLS,
    clean_matches,
    load_ranking,
    load_world_cup,
    build_feature_matrix,
)

HISTORICAL_PATH = "data/raw/matches_1930_2022.csv"
MATCHES_2026_PATH = "data/processed/matches_2026.csv"
SCHEDULE_PATH = "data/raw/schedule_2026.csv"
TEST_FROM_YEAR = 2018

# Synthetic scores only encode the 1X2 outcome (the model never sees goals).
RESULT_TO_SCORES = {"home_win": (1, 0), "draw": (0, 0), "away_win": (0, 1)}


def lookup_round(schedule: pd.DataFrame, date: str, home: str, away: str) -> str:
    row = schedule[
        (schedule["Date"] == date)
        & (schedule["home_team"] == home)
        & (schedule["away_team"] == away)
    ]
    if not row.empty:
        return row.iloc[0]["Round"]
    return "Group stage"


def collect_2026_matches(results: dict, schedule: pd.DataFrame) -> pd.DataFrame:
    """Completed 2026 matches as match rows, deduplicated by (date, home, away).

    Written to a separate file — the historical CSV is the immutable source
    dataset and must never be rewritten (it is gitignored, so corruption
    would be unrecoverable; the old in-place append also duplicated rows on
    every rerun).
    """
    rows = {}
    for m in results["matches"]:
        # Include --no-prediction entries (correct=None) — actual_result is all we need.
        if m["actual_result"] is None:
            continue
        h_score, a_score = RESULT_TO_SCORES[m["actual_result"]]
        key = (m["date"], m["home_team"], m["away_team"])
        rows[key] = {
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_score": h_score,
            "away_score": a_score,
            "Round": lookup_round(schedule, m["date"], m["home_team"], m["away_team"]),
            "Date": m["date"],
            "Year": 2026,
        }
    return pd.DataFrame(list(rows.values()))


def main():
    results_path = "results/results.json"
    if not os.path.exists(results_path):
        print("ERROR: results/results.json not found.")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    schedule = pd.read_csv(SCHEDULE_PATH)
    new_df = collect_2026_matches(results, schedule)
    if new_df.empty:
        print("No completed matches to add. Nothing to retrain.")
        sys.exit(0)

    new_df.to_csv(MATCHES_2026_PATH, index=False)
    print(f"Found {len(new_df)} completed 2026 match(es) -> {MATCHES_2026_PATH}")

    original_df = pd.read_csv(HISTORICAL_PATH)
    combined = pd.concat(
        [original_df, new_df.reindex(columns=original_df.columns)],
        ignore_index=True,
    )
    print(f"Training set: {len(original_df)} historical + {len(new_df)} from 2026")

    df = clean_matches(combined)
    ranking_df = load_ranking("data/raw/fifa_ranking_2026-06-08.csv")
    world_cup_df = load_world_cup()
    features_df = build_feature_matrix(df, world_cup_df, ranking_df)

    X = features_df[FEATURE_COLS].values
    y = features_df["target"].values
    sample_weights = np.where(features_df["Year"] >= 2010, 2.0, 1.0)

    test_mask = features_df["Year"].values >= TEST_FROM_YEAR
    X_train, y_train, w_train = X[~test_mask], y[~test_mask], sample_weights[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    old_model = joblib.load("models/model.pkl")
    params = old_model.get_params()
    params.pop("n_jobs", None)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"\nRetrained XGBoost metrics (holdout {TEST_FROM_YEAR}+, incl. 2026 results):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Log Loss:  {ll:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["home_win", "draw", "away_win"]))

    # Final model is fit on ALL data (incl. holdout) — evaluation above is
    # for monitoring only; at deployment we want every available match.
    final_model = XGBClassifier(**params)
    final_model.fit(X, y, sample_weight=sample_weights)
    joblib.dump(final_model, "models/model.pkl")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
