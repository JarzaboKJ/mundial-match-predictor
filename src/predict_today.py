import argparse
import html
import sys
import os
import json
from datetime import date

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib

from preprocessing import (
    STAGE_MAP,
    FEATURE_COLS,
    canonical_name,
    load_and_clean_matches,
    load_ranking,
    load_world_cup,
    build_team_stats,
    build_h2h_stats,
    build_single_match_features,
)

# Weight of the (adjusted) model when bookmaker odds are available.
# Injury/form/morale corrections are applied to the MODEL probabilities
# BEFORE blending: bookmakers already price news into their odds, so
# adjusting the blended number double-counts the same information.
BLEND_RATIO = 0.15
CLASS_NAMES = ["home_win", "draw", "away_win"]


def esc(value):
    return html.escape(str(value), quote=True)


def parse_morale(morale_text):
    text = str(morale_text or "").strip().lower()
    level = text.split()[0] if text.split() else ""
    if level == "high":
        return 0.03
    elif level == "low":
        return -0.03
    return 0.0


def safe_form_score(value):
    try:
        return max(-2, min(2, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def apply_injury_adjustment(proba, home_injuries, away_injuries):
    """Scale each side's win probability down by its injury impact.

    Multiplicative (p *= 1 - impact) instead of subtractive with a
    p > impact gate: the old gate silently skipped the adjustment whenever
    a side's win probability was below its injury impact (e.g. underdogs
    with several injuries got no penalty at all).
    """
    p = proba.copy()

    home_impact = sum(inj.get("importance", 5) * 0.008 for inj in home_injuries)
    away_impact = sum(inj.get("importance", 5) * 0.008 for inj in away_injuries)

    home_impact = min(home_impact, 0.15)
    away_impact = min(away_impact, 0.15)

    if home_impact > 0:
        removed = p[0] * home_impact
        p[0] -= removed
        other = p[1] + p[2]
        if other > 0:
            p[1] += removed * (p[1] / other)
            p[2] += removed * (p[2] / other)

    if away_impact > 0:
        removed = p[2] * away_impact
        p[2] -= removed
        other = p[0] + p[1]
        if other > 0:
            p[0] += removed * (p[0] / other)
            p[1] += removed * (p[1] / other)

    p = np.clip(p, 0.01, None)
    return p / p.sum(), home_impact, away_impact


def apply_form_adjustment(proba, home_form_score, away_form_score):
    p = proba.copy()
    form_adj = (home_form_score - away_form_score) * 0.02
    p[0] += form_adj
    p[2] -= form_adj
    p = np.clip(p, 0.01, None)
    return p / p.sum(), form_adj


def apply_morale_adjustment(proba, home_morale_text, away_morale_text):
    p = proba.copy()
    home_adj = parse_morale(home_morale_text)
    away_adj = parse_morale(away_morale_text)

    p[0] += home_adj
    p[2] += away_adj

    p = np.clip(p, 0.01, None)
    return p / p.sum(), home_adj, away_adj


def valid_bookie_proba(cur):
    """Implied probabilities from the fetch step, or None if unusable.

    Legacy files stored (0, 0, 0) when no odds were found; treating that as
    a real distribution made the blend divide by zero and emit NaNs.
    """
    if cur is None:
        return None
    if cur.get("has_bookie_odds") is False:
        return None
    p = np.array([
        float(cur.get("bookie_home_prob") or 0.0),
        float(cur.get("bookie_draw_prob") or 0.0),
        float(cur.get("bookie_away_prob") or 0.0),
    ])
    if not np.all(np.isfinite(p)) or p.sum() < 0.5:
        return None
    return p / p.sum()


def proba_dict(p):
    return {
        "home_win": round(float(p[0]), 4),
        "draw": round(float(p[1]), 4),
        "away_win": round(float(p[2]), 4),
    }


def generate_html(predictions, today_str):
    os.makedirs("outputs", exist_ok=True)
    rows_html = ""
    bench_rows = ""

    for pred in predictions:
        h = esc(pred["home_team"])
        a = esc(pred["away_team"])
        fp = pred["final_proba"]
        mp = pred["model_proba"]
        bp = pred["bookie_proba"]
        ap = pred["adjusted_model_proba"]

        fh, fd, fa = fp["home_win"] * 100, fp["draw"] * 100, fp["away_win"] * 100
        mh, md, ma = mp["home_win"] * 100, mp["draw"] * 100, mp["away_win"] * 100
        ah, ad, aa = ap["home_win"] * 100, ap["draw"] * 100, ap["away_win"] * 100

        fav_idx = np.argmax([fh, fd, fa])
        fav_labels = [f"{h} wygra", "Remis", f"{a} wygra"]
        fav_label = fav_labels[fav_idx]
        fav_pct = [fh, fd, fa][fav_idx]

        home_injuries = pred.get("home_injuries", [])
        away_injuries = pred.get("away_injuries", [])
        home_morale = pred.get("home_morale", "medium")
        away_morale = pred.get("away_morale", "medium")
        home_form_score = pred.get("home_form_score", 0)
        away_form_score = pred.get("away_form_score", 0)
        injury_home = pred.get("injury_impact_home", 0)
        injury_away = pred.get("injury_impact_away", 0)
        form_adj = pred.get("form_adjustment", 0)
        morale_home = pred.get("morale_adjustment_home", 0)
        morale_away = pred.get("morale_adjustment_away", 0)

        inj_html = ""
        if home_injuries:
            items = "".join(
                f'<li>{esc(p["player"])} <span class="imp">(importance={esc(p.get("importance", "?"))})</span></li>'
                for p in home_injuries
            )
            inj_html += f'<div class="inj-list"><strong>{h}:</strong><ul>{items}</ul></div>'
        else:
            inj_html += f'<div class="inj-list"><strong>{h}:</strong> brak kontuzji</div>'

        if away_injuries:
            items = "".join(
                f'<li>{esc(p["player"])} <span class="imp">(importance={esc(p.get("importance", "?"))})</span></li>'
                for p in away_injuries
            )
            inj_html += f'<div class="inj-list"><strong>{a}:</strong><ul>{items}</ul></div>'
        else:
            inj_html += f'<div class="inj-list"><strong>{a}:</strong> brak kontuzji</div>'

        morale_net = (morale_home - morale_away) * 100
        adj_html = f"""
            <div class="adjustments">
                <h4>Adjustmenty (na komponencie modelowym, przed blendem)</h4>
                <div class="adj-row">
                    <span class="adj-label">Injury penalty:</span>
                    <span>{h} {f'-{injury_home*100:.1f}%' if injury_home > 0 else '0%'} | {a} {f'-{injury_away*100:.1f}%' if injury_away > 0 else '0%'}</span>
                </div>
                <div class="adj-row">
                    <span class="adj-label">Forma:</span>
                    <span>{h} {home_form_score:+d} | {a} {away_form_score:+d} &rarr; form adjustment {form_adj*100:+.0f}%</span>
                </div>
                <div class="adj-row">
                    <span class="adj-label">Morale:</span>
                    <span>{esc(home_morale)} vs {esc(away_morale)} &rarr; morale adjustment {morale_net:+.0f}%</span>
                </div>
            </div>"""

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
                <div class="injuries-detail">{inj_html}</div>
                {adj_html}
            </div>
        </div>
        """

        if bp is not None:
            bookie_cells = (
                f'<td>{bp["home_win"]*100:.1f}%</td><td>{bp["draw"]*100:.1f}%</td>'
                f'<td>{bp["away_win"]*100:.1f}%</td>'
            )
        else:
            bookie_cells = '<td colspan="3">brak kursów</td>'

        bench_rows += f"""
        <tr>
            <td rowspan="4" class="match-cell">{h} vs {a}</td>
            <td>Model</td>
            <td>{mh:.1f}%</td><td>{md:.1f}%</td><td>{ma:.1f}%</td>
        </tr>
        <tr>
            <td>Model + korekty</td>
            <td>{ah:.1f}%</td><td>{ad:.1f}%</td><td>{aa:.1f}%</td>
        </tr>
        <tr>
            <td>Bukmacherzy</td>
            {bookie_cells}
        </tr>
        <tr class="final-row">
            <td>Finalna</td>
            <td>{fh:.1f}%</td><td>{fd:.1f}%</td><td>{fa:.1f}%</td>
        </tr>
        """

    html_doc = f"""<!DOCTYPE html>
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
        margin-bottom: 0.8rem;
        font-size: 0.85rem;
        color: #ccc;
    }}
    .inj-list {{ margin-bottom: 0.4rem; }}
    .inj-list ul {{ margin-left: 1.5rem; }}
    .inj-list li {{ margin-bottom: 0.2rem; }}
    .imp {{ color: #f39c12; font-size: 0.8rem; }}
    .adjustments {{
        margin-top: 0.8rem;
        padding: 0.8rem 1rem;
        background: rgba(15, 52, 96, 0.5);
        border-radius: 8px;
    }}
    .adjustments h4 {{
        color: #e94560;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }}
    .adj-row {{
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
    }}
    .adj-label {{
        color: #8899aa;
        margin-right: 0.5rem;
    }}
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
<h3>Benchmark: Model vs Model+korekty vs Bukmacherzy vs Finalna</h3>
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

    path = f"outputs/daily_predictions_{today_str}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_doc)
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


def upsert_result(results_data, entry):
    """Insert the prediction, or refresh it if the match has no result yet.

    Predictions whose actual_result is already recorded are immutable —
    rewriting them after the fact would falsify the accuracy tracking.
    """
    for existing in results_data["matches"]:
        if (
            existing["home_team"] == entry["home_team"]
            and existing["away_team"] == entry["away_team"]
            and existing["date"] == entry["date"]
        ):
            if existing.get("actual_result") is None:
                existing.update(entry)
            return
    results_data["matches"].append(entry)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for a match day")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Match day in ISO format (default: today)",
    )
    args = parser.parse_args()
    today_str = args.date

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

    print("Loading model...")
    xgb_model = joblib.load("models/model.pkl")

    print("Building historical stats...")
    df = load_and_clean_matches()
    world_cup_df = load_world_cup()
    team_stats = build_team_stats(df, world_cup_df, as_of_year=2026)
    h2h_stats = build_h2h_stats(df)
    ranking_df = load_ranking("data/raw/fifa_ranking_2026-06-08.csv")

    results_data = load_or_create_results()

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
            cur = current_by_match.get((canonical_name(home), canonical_name(away)))

        bookie_proba = valid_bookie_proba(cur)
        if cur:
            home_injuries = cur.get("home_injuries") or []
            away_injuries = cur.get("away_injuries") or []
            home_form_score = safe_form_score(cur.get("home_form_score"))
            away_form_score = safe_form_score(cur.get("away_form_score"))
            home_morale = str(cur.get("home_morale") or "medium")
            away_morale = str(cur.get("away_morale") or "medium")
        else:
            home_injuries = []
            away_injuries = []
            home_form_score = 0
            away_form_score = 0
            home_morale = "medium"
            away_morale = "medium"

        # 1) Adjust the model with match-day news (the model can't know it).
        adjusted, home_impact, away_impact = apply_injury_adjustment(
            model_proba, home_injuries, away_injuries
        )
        adjusted, form_adj = apply_form_adjustment(
            adjusted, home_form_score, away_form_score
        )
        adjusted, morale_adj_home, morale_adj_away = apply_morale_adjustment(
            adjusted, home_morale, away_morale
        )

        # 2) Blend with bookmakers; without odds, the adjusted model stands alone.
        if bookie_proba is not None:
            final_proba = BLEND_RATIO * adjusted + (1 - BLEND_RATIO) * bookie_proba
            final_proba = final_proba / final_proba.sum()
        else:
            final_proba = adjusted

        winner_idx = int(np.argmax(final_proba))
        predicted_winner = CLASS_NAMES[winner_idx]

        h2h_hw = int(features["h2h_home_wins"].iloc[0])
        h2h_d = int(features["h2h_draws"].iloc[0])
        h2h_aw = int(features["h2h_away_wins"].iloc[0])
        has_h2h = (h2h_hw + h2h_d + h2h_aw) > 0

        morale_net = (morale_adj_home - morale_adj_away) * 100

        print(f"\n{'='*55}")
        print(f"  === {home} vs {away} ===")
        print(f"{'='*55}")

        if home_injuries:
            inj_str = ", ".join(
                f'{p["player"]} (importance={p.get("importance", "?")})' for p in home_injuries
            )
            print(f"  Kontuzje: {home} → {inj_str}")
        else:
            print(f"  Kontuzje: {home} → brak")
        if away_injuries:
            inj_str = ", ".join(
                f'{p["player"]} (importance={p.get("importance", "?")})' for p in away_injuries
            )
            print(f"            {away} → {inj_str}")
        else:
            print(f"            {away} → brak")

        print(f"  Injury penalty: {home} -{home_impact*100:.1f}% | {away} -{away_impact*100:.1f}%")
        print(f"  Forma: {home} {home_form_score:+d} | {away} {away_form_score:+d}"
              f" → form adjustment {form_adj*100:+.0f}%")
        print(f"  Morale: {home_morale} vs {away_morale}"
              f" → morale adjustment {morale_net:+.0f}%")

        print(f"  Model:         Home {model_proba[0]*100:4.0f}% | Draw {model_proba[1]*100:4.0f}% | Away {model_proba[2]*100:4.0f}%")
        print(f"  Model+korekty: Home {adjusted[0]*100:4.0f}% | Draw {adjusted[1]*100:4.0f}% | Away {adjusted[2]*100:4.0f}%")
        if bookie_proba is not None:
            print(f"  Bukmacherzy:   Home {bookie_proba[0]*100:4.0f}% | Draw {bookie_proba[1]*100:4.0f}% | Away {bookie_proba[2]*100:4.0f}%")
        else:
            print("  Bukmacherzy:   brak kursów → finalna = model + korekty")
        print(f"  FINALNA:       Home {final_proba[0]*100:4.0f}% | Draw {final_proba[1]*100:4.0f}% | Away {final_proba[2]*100:4.0f}%")
        winner_name = home if winner_idx == 0 else (away if winner_idx == 2 else "REMIS")
        print(f"  Predykcja: {winner_name} wygra ({final_proba[winner_idx]*100:.0f}%)")
        h2h_tag = f"H2H: {h2h_hw}-{h2h_d}-{h2h_aw}" if has_h2h else "H2H: brak"
        print(f"  {h2h_tag}")

        pred = {
            "date": today_str,
            "home_team": home,
            "away_team": away,
            "model_proba": proba_dict(model_proba),
            "adjusted_model_proba": proba_dict(adjusted),
            "bookie_proba": proba_dict(bookie_proba) if bookie_proba is not None else None,
            "final_proba": proba_dict(final_proba),
            "predicted_winner": predicted_winner,
            "actual_result": None,
            "correct": None,
            "home_injuries": home_injuries,
            "away_injuries": away_injuries,
            "home_injuries_count": len(home_injuries),
            "away_injuries_count": len(away_injuries),
            "home_form_score": home_form_score,
            "away_form_score": away_form_score,
            "home_morale": home_morale,
            "away_morale": away_morale,
            "injury_impact_home": round(float(home_impact), 4),
            "injury_impact_away": round(float(away_impact), 4),
            "form_adjustment": round(float(form_adj), 4),
            "morale_adjustment_home": round(float(morale_adj_home), 4),
            "morale_adjustment_away": round(float(morale_adj_away), 4),
        }
        predictions.append(pred)

        upsert_result(results_data, {
            "date": today_str,
            "home_team": home,
            "away_team": away,
            "model_proba": pred["model_proba"],
            "adjusted_model_proba": pred["adjusted_model_proba"],
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
