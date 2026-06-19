import os
import json
import re
import sys
import time
from datetime import date

import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()

SCHEDULE_PATH = "data/raw/schedule_2026.csv"
OUTPUT_DIR = "data/processed"
MODEL = "claude-haiku-4-5-20251001"
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}
MAX_PAUSE_TURNS = 5

PROMPT_TEMPLATE = """Today is {date}. The FIFA World Cup 2026 match {home} vs {away} is scheduled for today.

Search the web for the latest pre-match information and return a JSON object with exactly this structure (no markdown fences, no extra text — only valid JSON):

{{
  "home_team": "{home}",
  "away_team": "{away}",
  "home_injuries": [
    {{"player": "<name>", "importance": <1-10>, "role": "<role>"}}
  ],
  "away_injuries": [
    {{"player": "<name>", "importance": <1-10>, "role": "<role>"}}
  ],
  "home_form_score": <-2 to +2>,
  "away_form_score": <-2 to +2>,
  "home_form": "<last 5 results, e.g. W-W-D-L-W>",
  "away_form": "<last 5 results>",
  "home_morale": "<high|medium|low>",
  "away_morale": "<high|medium|low>",
  "bookmaker_odds": [
    {{
      "source": "<bookmaker name>",
      "home_win": <decimal odds>,
      "draw": <decimal odds>,
      "away_win": <decimal odds>
    }}
  ],
  "key_news": ["<headline 1>", "<headline 2>"]
}}

Injury importance scale (rate each injured/doubtful player):
- 8-10: captain, top scorer, only player at position, irreplaceable starter
- 5-7: regular starter, important squad member
- 1-4: backup, rotation player, reserve
Role: one of key_scorer, key_defender, key_midfielder, key_winger, captain, goalkeeper, squad_player, backup.
Only list players who are injured, doubtful, or suspended — not fully fit players.

form_score scale based on last 5 matches:
- +2: excellent (4-5 wins, dominant form)
- +1: good (3 wins or strong recent run)
-  0: average/mixed results
- -1: poor (mostly draws and losses)
- -2: terrible (0-1 wins, losing streak)

Team name aliases — when searching for odds or news, try both names:
- "Côte d'Ivoire" = "Ivory Coast"
- "Türkiye" = "Turkey"
- "Korea Republic" = "South Korea"
- "IR Iran" = "Iran"
- "Congo DR" = "DR Congo" = "Democratic Republic of Congo"
- "Bosnia-Herzegovina" = "Bosnia and Herzegovina" = "Bosnia"
- "Cape Verde" = "Cabo Verde"
- "Curaçao" = "Curacao"
- "Czechia" = "Czech Republic"
- "United States" = "USA" = "USMNT"

If you cannot find bookmaker odds under the exact team name, try alternative names:
Côte d'Ivoire/Ivory Coast, Cabo Verde/Cape Verde, IR Iran/Iran,
DR Congo/Congo DR/DRC, Korea Republic/South Korea.
Always search at least 2 query variations before reporting no odds found."""


def get_todays_matches():
    df = pd.read_csv(SCHEDULE_PATH)
    today = date.today().isoformat()
    matches = df[df["Date"] == today]
    return matches[["home_team", "away_team"]].to_dict("records")


def _api_call_with_retry(client, **kwargs):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
        ) as e:
            wait = 60 * (attempt + 1)
            print(f"    {type(e).__name__}, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError("Max retries exceeded for Anthropic API call")


def fetch_match_data(client, home, away):
    today_str = date.today().isoformat()
    prompt = PROMPT_TEMPLATE.format(date=today_str, home=home, away=away)

    response = _api_call_with_retry(
        client,
        model=MODEL,
        max_tokens=4096,
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )

    pause_turns = 0
    while response.stop_reason == "pause_turn" and pause_turns < MAX_PAUSE_TURNS:
        pause_turns += 1
        response = _api_call_with_retry(
            client,
            model=MODEL,
            max_tokens=4096,
            tools=[WEB_SEARCH_TOOL],
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response.content},
            ],
        )

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:300]}")
    return json.loads(match.group())


def _valid_odds_entry(odds):
    """Decimal 1X2 odds are only usable if all three are numbers > 1.0."""
    if not isinstance(odds, dict):
        return False
    for key in ("home_win", "draw", "away_win"):
        value = odds.get(key)
        if not isinstance(value, (int, float)) or value <= 1.0:
            return False
    return True


def compute_implied_probabilities(odds_list):
    """Average implied probabilities across bookmakers, margin removed.

    Returns None if there is no usable odds entry — callers must treat that
    as 'no bookmaker signal', not as zero probabilities.
    """
    valid = [o for o in (odds_list or []) if _valid_odds_entry(o)]
    if not valid:
        return None

    raw_home = sum(1.0 / o["home_win"] for o in valid) / len(valid)
    raw_draw = sum(1.0 / o["draw"] for o in valid) / len(valid)
    raw_away = sum(1.0 / o["away_win"] for o in valid) / len(valid)

    total = raw_home + raw_draw + raw_away
    return raw_home / total, raw_draw / total, raw_away / total


def _clamp_int(value, lo, hi, default=0):
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _norm_morale(value):
    text = str(value or "").strip().lower()
    first = text.split()[0] if text.split() else ""
    return first if first in ("high", "medium", "low") else "medium"


def _sanitize_injuries(injuries):
    clean = []
    for inj in injuries or []:
        if not isinstance(inj, dict):
            continue
        clean.append({
            "player": str(inj.get("player", "unknown")),
            "importance": _clamp_int(inj.get("importance"), 1, 10, default=5),
            "role": str(inj.get("role", "unknown")),
        })
    return clean


def build_match_record(data, home, away):
    """Validate and normalize the LLM's JSON — it is untrusted input.

    Team names come from the schedule (not from the model output) so that
    predict_today.py can always join records back to fixtures.
    """
    odds_list = data.get("bookmaker_odds", [])
    implied = compute_implied_probabilities(odds_list)
    has_odds = implied is not None
    bookie_home, bookie_draw, bookie_away = implied if has_odds else (0.0, 0.0, 0.0)

    home_injuries = _sanitize_injuries(data.get("home_injuries"))
    away_injuries = _sanitize_injuries(data.get("away_injuries"))

    return {
        "home_team": home,
        "away_team": away,
        "home_injuries": home_injuries,
        "away_injuries": away_injuries,
        "home_injuries_count": len(home_injuries),
        "away_injuries_count": len(away_injuries),
        "home_form": str(data.get("home_form") or ""),
        "away_form": str(data.get("away_form") or ""),
        "home_form_score": _clamp_int(data.get("home_form_score"), -2, 2),
        "away_form_score": _clamp_int(data.get("away_form_score"), -2, 2),
        "home_morale": _norm_morale(data.get("home_morale")),
        "away_morale": _norm_morale(data.get("away_morale")),
        "has_bookie_odds": has_odds,
        "bookie_home_prob": round(bookie_home, 4),
        "bookie_draw_prob": round(bookie_draw, 4),
        "bookie_away_prob": round(bookie_away, 4),
        "raw_odds": odds_list,
        "key_news": [str(n) for n in data.get("key_news") or []],
    }


def print_summary(record):
    h, a = record["home_team"], record["away_team"]
    print(f"\n{'='*60}")
    print(f"  {h}  vs  {a}")
    print(f"{'='*60}")

    home_inj = record.get("home_injuries", [])
    away_inj = record.get("away_injuries", [])
    if home_inj:
        inj_str = ", ".join(f'{p["player"]} (imp={p["importance"]})' for p in home_inj)
        print(f"  {h} injuries: {inj_str}")
    else:
        print(f"  {h} injuries: brak")
    if away_inj:
        inj_str = ", ".join(f'{p["player"]} (imp={p["importance"]})' for p in away_inj)
        print(f"  {a} injuries: {inj_str}")
    else:
        print(f"  {a} injuries: brak")

    print(f"  Form: {h} {record.get('home_form', '')} ({record.get('home_form_score', 0):+d})"
          f"  |  {a} {record.get('away_form', '')} ({record.get('away_form_score', 0):+d})")
    print(f"  Morale: {h}: {record['home_morale']}  |  {a}: {record['away_morale']}")
    if record.get("has_bookie_odds"):
        print(f"  Implied probs — H: {record['bookie_home_prob']:.2%}  "
              f"D: {record['bookie_draw_prob']:.2%}  "
              f"A: {record['bookie_away_prob']:.2%}")
    else:
        print("  Implied probs — brak kursów (model-only fallback)")
    if record["key_news"]:
        print("  Key news:")
        for n in record["key_news"]:
            print(f"    * {n}")


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in .env")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    matches = get_todays_matches()

    if not matches:
        print(f"No matches scheduled for today ({date.today().isoformat()}).")
        sys.exit(0)

    print(f"Found {len(matches)} match(es) for {date.today().isoformat()}")

    records = []
    for match in matches:
        home, away = match["home_team"], match["away_team"]
        if records:
            print("  (waiting 60s for rate limit cooldown...)")
            time.sleep(60)
        print(f"\nFetching data for {home} vs {away}...")
        try:
            raw_data = fetch_match_data(client, home, away)
            record = build_match_record(raw_data, home, away)
            records.append(record)
            print_summary(record)
        except Exception as e:
            print(f"  ERROR fetching {home} vs {away}: {e}")

    if records:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"current_features_{date.today().isoformat()}.json")
        output = {"date": date.today().isoformat(), "matches": records}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
