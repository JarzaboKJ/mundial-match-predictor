import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier

from preprocessing import (
    FEATURE_COLS,
    load_and_clean_matches,
    build_team_stats,
    build_h2h_stats,
    build_feature_matrix,
)

RESULT_TO_TARGET = {"home_win": 0, "draw": 1, "away_win": 2}


def main():
    results_path = "results/results.json"
    if not os.path.exists(results_path):
        print("ERROR: results/results.json not found.")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    completed = [m for m in results["matches"] if m["actual_result"] is not None]
    if not completed:
        print("No completed matches to add. Nothing to retrain.")
        sys.exit(0)

    print(f"Found {len(completed)} completed match(es) to incorporate.")

    matches_path = "data/raw/matches_1930_2022.csv"
    original_df = pd.read_csv(matches_path)

    new_rows = []
    for m in completed:
        target_map = {"home_win": (1, 0), "draw": (0, 0), "away_win": (0, 1)}
        h_score, a_score = target_map[m["actual_result"]]
        new_rows.append({
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_score": h_score,
            "away_score": a_score,
            "Round": "Group stage",
            "Date": m["date"],
            "Year": 2026,
        })

    new_df = pd.DataFrame(new_rows)
    for col in original_df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    combined = pd.concat([original_df, new_df[original_df.columns]], ignore_index=True)
    combined.to_csv(matches_path, index=False)
    print(f"Dataset updated: {len(original_df)} -> {len(combined)} matches")

    df = load_and_clean_matches(matches_path)
    team_stats = build_team_stats(df)
    h2h_stats = build_h2h_stats(df)
    ranking_df = pd.read_csv("data/raw/fifa_ranking_2026-06-08.csv")
    features_df = build_feature_matrix(df, team_stats, h2h_stats, ranking_df)

    X = features_df[FEATURE_COLS].values
    y = features_df["target"].values
    sample_weights = np.where(features_df["Year"] >= 2010, 2.0, 1.0)

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )

    old_model = joblib.load("models/model.pkl")
    params = old_model.get_params()
    params.pop("n_jobs", None)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"\nRetrained XGBoost metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Log Loss:  {ll:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["home_win", "draw", "away_win"]))

    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    main()
