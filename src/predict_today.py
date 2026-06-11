import sys
import os
import json
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib

from preprocessing import (
    NAME_MAP,
    STAGE_MAP,
    FEATURE_COLS,
    load_and_clean_matches,
    build_team_stats,
    build_h2h_stats,
    build_single_match_features,
)

BLEND_RATIO = 0.35
CLASS_NAMES = ["home_win", "draw", "away_win"]


def parse_morale(morale_text):
    level = morale_text.strip().split()[0].lower()
    if level == "high":
        return 0.03
    elif level == "low":
        return -0.03
    return 0.0


def apply_injury_adjustment(proba, home_inj, away_inj):
    p = proba.copy()
    home_penalty = 0.02 * min(home_inj, 3)
    away_penalty = 0.02 * min(away_inj, 3)

    if home_penalty > 0 and p[0] > home_penalty:
        p[0] -= home_penalty
        p[2] += home_penalty

    if away_penalty > 0 and p[2] > away_penalty:
        p[2] -= away_penalty
        p[0] += away_penalty

    return p / p.sum()


def apply_morale_adjustment(proba, home_morale_text, away_morale_text):
    p = proba.copy()
    home_adj = parse_morale(home_morale_text)
    away_adj = parse_morale(away_morale_text)

    p[0] += home_adj
    p[2] += away_adj

    p = np.clip(p, 0.01, None)
    return p / p.sum()


def generate_html(predictions, today_str):
    os.makedirs("outputs", exist_ok=True)
    rows_html = ""
    bench_rows = ""

    for pred in predictions:
        h = pred["home_team"]
        a = pred["away_team"]
        fp = pred["final_proba"]
        mp = pred["model_proba"]
        bp = pred["bookie_proba"]

        fh, fd, fa = fp["home_win"] * 100, fp["draw"] * 100, fp["away_win"] * 100
        mh, md, ma = mp["home_win"] * 100, mp["draw"] * 100, mp["away_win"] * 100
        bh, bd, ba = bp["home_win"] * 100, bp["draw"] * 100, bp["away_win"] * 100

        fav_idx = np.argmax([fh, fd, fa])
        fav_labels = [f"{h} wygra", "Remis", f"{a} wygra"]
        fav_label = fav_labels[fav_idx]
        fav_pct = [fh, fd, fa][fav_idx]

        home_inj = pred.get("home_injuries_count", 0)
        away_inj = pred.get("away_injuries_count", 0)
        home_morale = pred.get("home_morale", "medium")
        away_morale = pred.get("away_morale", "medium")

        home_inj_players = pred.get("home_injuries_players", [])
        away_inj_players = pred.get("away_injuries_players", [])

        inj_html = ""
        if home_inj_players:
            items = "".join(f"<li>{p}</li>" for p in home_inj_players)
            inj_html += f'<div class="inj-list"><strong>{h}:</strong><ul>{items}</ul></div>'
        if away_inj_players:
            items = "".join(f"<li>{p}</li>" for p in away_inj_players)
            inj_html += f'<div class="inj-list"><strong>{a}:</strong><ul>{items}</ul></div>'

        rows_html += f"""
        <div class="match-card">
            <h2>{h} vs {a}</h2>
            <div class="prediction-badge">PREDYKCJA: {fav_label} ({fav_pct:.0f}%)</div>
            <div class="bars">
                <div class="bar-row">
                    <span class="bar-label">{h}</span>
                    <div class="bar-track">
                        <div class="bar bar-home" style="width:{fh:.1f}%">
                            <span>{fh:.0f}%</span>
                        </div>
                    </div>
                </div>
                <div class="bar-row">
                    <span class="bar-label">Remis</span>
                    <div class="bar-track">
                        <div class="bar bar-draw" style="width:{fd:.1f}%">
                            <span>{fd:.0f}%</span>
                        </div>
                    </div>
                </div>
                <div class="bar-row">
                    <span class="bar-label">{a}</span>
                    <div class="bar-track">
                        <div class="bar bar-away" style="width:{fa:.1f}%">
                            <span>{fa:.0f}%</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="details">
                <div class="detail-row">
                    <span class="detail-label">Kontuzje:</span>
                    <span>{h} = {home_inj}, {a} = {away_inj}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Morale:</span>
                    <span>{h}: {home_morale.split()[0]} | {a}: {away_morale.split()[0]}</span>
                </div>
                {f'<div class="injuries-detail">{inj_html}</div>' if inj_html else ''}
            </div>
        </div>
        """

        bench_rows += f"""
        <tr>
            <td rowspan="3" class="match-cell">{h} vs {a}</td>
            <td>Model</td>
            <td>{mh:.1f}%</td><td>{md:.1f}%</td><td>{ma:.1f}%</td>
        </tr>
        <tr>
            <td>Bukmacherzy</td>
            <td>{bh:.1f}%</td><td>{bd:.1f}%</td><td>{ba:.1f}%</td>
        </tr>
        <tr class="final-row">
            <td>Finalna</td>
            <td>{fh:.1f}%</td><td>{fd:.1f}%</td><td>{fa:.1f}%</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mundial 2026 — Predykcje na {today_str}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: #1a1a2e;
        color: #e0e0e0;
        font-family: 'Segoe UI', system-ui, sans-serif;
        padding: 2rem;
    }}
    h1 {{
        text-align: center;
        color: #e94560;
        font-size: 2rem;
        margin-bottom: 2rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    .match-card {{
        background: #16213e;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #0f3460;
    }}
    .match-card h2 {{
        color: #e94560;
        font-size: 1.4rem;
        margin-bottom: 0.8rem;
    }}
    .prediction-badge {{
        background: #e94560;
        color: white;
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }}
    .bars {{ margin: 1rem 0; }}
    .bar-row {{
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }}
    .bar-label {{
        width: 140px;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    .bar-track {{
        flex: 1;
        background: #0f3460;
        border-radius: 6px;
        height: 28px;
        overflow: hidden;
    }}
    .bar {{
        height: 100%;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.85rem;
        min-width: 40px;
        color: #1a1a2e;
    }}
    .bar-home {{ background: linear-gradient(90deg, #27ae60, #2ecc71); }}
    .bar-draw {{ background: linear-gradient(90deg, #f39c12, #f1c40f); }}
    .bar-away {{ background: linear-gradient(90deg, #c0392b, #e74c3c); }}
    .details {{
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #0f3460;
    }}
    .detail-row {{
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }}
    .detail-label {{
        color: #8899aa;
        margin-right: 0.5rem;
    }}
    .injuries-detail {{
        margin-top: 0.8rem;
        font-size: 0.85rem;
        color: #aaa;
    }}
    .inj-list {{ margin-bottom: 0.4rem; }}
    .inj-list ul {{ margin-left: 1.5rem; }}
    .inj-list li {{ margin-bottom: 0.2rem; }}
    h3 {{
        text-align: center;
        color: #e94560;
        margin: 2rem 0 1rem;
        font-size: 1.3rem;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        background: #16213e;
        border-radius: 8px;
        overflow: hidden;
    }}
    th {{
        background: #0f3460;
        padding: 0.7rem;
        text-align: center;
        color: #e94560;
        font-size: 0.9rem;
    }}
    td {{
        padding: 0.5rem 0.7rem;
        text-align: center;
        border-bottom: 1px solid #0f3460;
        font-size: 0.9rem;
    }}
    .match-cell {{
        text-align: left;
        font-weight: bold;
        color: #e0e0e0;
    }}
    .final-row td {{
        background: rgba(233, 69, 96, 0.15);
        font-weight: bold;
    }}
</style>
</head>
<body>
<h1>Mundial 2026 — Predykcje na {today_str}</h1>
{rows_html}
<h3>Benchmark: Model vs Bukmacherzy vs Finalna predykcja</h3>
<table>
    <thead>
        <tr>
            <th>Mecz</th><th>Zrodlo</th><th>Home Win</th><th>Draw</th><th>Away Win</th>
        </tr>
    </thead>
    <tbody>
        {bench_rows}
    </tbody>
</table>
</body>
</html>"""

    path = f"outputs/daily_predictions.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML saved to {path}")


def load_or_create_results():
    path = "results/results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "matches": [],
        "stats": {"total_predictions": 0, "correct": 0, "accuracy": 0.0},
    }


def save_results(data):
    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    today_str = date.today().isoformat()

    schedule = pd.read_csv("data/raw/schedule_2026.csv")
    todays_matches = schedule[schedule["Date"] == today_str]
    if todays_matches.empty:
        print(f"Brak meczy na {today_str}.")
        sys.exit(0)

    current_path = f"data/processed/current_features_{today_str}.json"
    if not os.path.exists(current_path):
        print("Uruchom najpierw: python src/fetch_current_data.py")
        sys.exit(1)

    with open(current_path, "r", encoding="utf-8") as f:
        current_data = json.load(f)
    current_by_match = {}
    for m in current_data["matches"]:
        key = (m["home_team"], m["away_team"])
        current_by_match[key] = m

    print("Loading models...")
    xgb_model = joblib.load("models/model.pkl")
    lr_model = joblib.load("models/baseline_model.pkl")

    print("Building historical stats...")
    df = load_and_clean_matches()
    team_stats = build_team_stats(df)
    h2h_stats = build_h2h_stats(df)
    ranking_df = pd.read_csv("data/raw/fifa_ranking_2026-06-08.csv")

    results_data = load_or_create_results()
    existing_keys = {
        (m["home_team"], m["away_team"], m["date"])
        for m in results_data["matches"]
    }

    predictions = []

    for _, row in todays_matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        stage_str = row["Round"]
        stage = STAGE_MAP.get(stage_str, 1)

        features = build_single_match_features(
            home, away, stage, team_stats, h2h_stats, ranking_df
        )
        X = features[FEATURE_COLS].values

        model_proba = xgb_model.predict_proba(X)[0]

        cur = current_by_match.get((home, away))
        if cur is None:
            home_mapped = NAME_MAP.get(home, home)
            away_mapped = NAME_MAP.get(away, away)
            cur = current_by_match.get((home_mapped, away_mapped))

        if cur:
            bookie_proba = np.array([
                cur["bookie_home_prob"],
                cur["bookie_draw_prob"],
                cur["bookie_away_prob"],
            ])
            home_inj = cur.get("home_injuries_count", 0)
            away_inj = cur.get("away_injuries_count", 0)
            home_morale = cur.get("home_morale", "medium")
            away_morale = cur.get("away_morale", "medium")
            home_inj_players = cur.get("home_injuries_players", [])
            away_inj_players = cur.get("away_injuries_players", [])
        else:
            bookie_proba = model_proba.copy()
            home_inj = away_inj = 0
            home_morale = away_morale = "medium"
            home_inj_players = away_inj_players = []

        adjusted = model_proba.copy()
        adjusted = apply_injury_adjustment(adjusted, home_inj, away_inj)
        adjusted = apply_morale_adjustment(adjusted, home_morale, away_morale)

        final_proba = BLEND_RATIO * adjusted + (1 - BLEND_RATIO) * bookie_proba
        final_proba = final_proba / final_proba.sum()

        winner_idx = np.argmax(final_proba)
        predicted_winner = CLASS_NAMES[winner_idx]

        print(f"\n{'='*50}")
        print(f"  {home} vs {away}")
        print(f"{'='*50}")
        print(f"  Model:       Home {model_proba[0]*100:4.0f}% | Draw {model_proba[1]*100:4.0f}% | Away {model_proba[2]*100:4.0f}%")
        print(f"  Bukmacherzy: Home {bookie_proba[0]*100:4.0f}% | Draw {bookie_proba[1]*100:4.0f}% | Away {bookie_proba[2]*100:4.0f}%")
        print(f"  FINALNA:     Home {final_proba[0]*100:4.0f}% | Draw {final_proba[1]*100:4.0f}% | Away {final_proba[2]*100:4.0f}%")
        winner_name = home if winner_idx == 0 else (away if winner_idx == 2 else "REMIS")
        print(f"  Predykcja: {winner_name} wygra ({final_proba[winner_idx]*100:.0f}%)")
        print(f"  Kontuzje: {home}={home_inj}, {away}={away_inj} | Morale: {home_morale.split()[0]} vs {away_morale.split()[0]}")

        pred = {
            "date": today_str,
            "home_team": home,
            "away_team": away,
            "model_proba": {
                "home_win": round(float(model_proba[0]), 4),
                "draw": round(float(model_proba[1]), 4),
                "away_win": round(float(model_proba[2]), 4),
            },
            "bookie_proba": {
                "home_win": round(float(bookie_proba[0]), 4),
                "draw": round(float(bookie_proba[1]), 4),
                "away_win": round(float(bookie_proba[2]), 4),
            },
            "final_proba": {
                "home_win": round(float(final_proba[0]), 4),
                "draw": round(float(final_proba[1]), 4),
                "away_win": round(float(final_proba[2]), 4),
            },
            "predicted_winner": predicted_winner,
            "actual_result": None,
            "correct": None,
            "home_injuries_count": home_inj,
            "away_injuries_count": away_inj,
            "home_injuries_players": home_inj_players,
            "away_injuries_players": away_inj_players,
            "home_morale": home_morale,
            "away_morale": away_morale,
        }
        predictions.append(pred)

        if (home, away, today_str) not in existing_keys:
            results_data["matches"].append({
                "date": today_str,
                "home_team": home,
                "away_team": away,
                "model_proba": pred["model_proba"],
                "bookie_proba": pred["bookie_proba"],
                "final_proba": pred["final_proba"],
                "predicted_winner": predicted_winner,
                "actual_result": None,
                "correct": None,
            })

    save_results(results_data)
    print(f"\nResults saved to results/results.json")

    generate_html(predictions, today_str)


if __name__ == "__main__":
    main()
