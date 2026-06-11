import pandas as pd
import numpy as np
from itertools import combinations

NAME_MAP = {
    "United States": "USA",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cabo Verde",
}

STAGE_MAP = {
    "Group stage": 1,
    "First group stage": 1,
    "Second group stage": 1,
    "Group stage play-off": 1,
    "First round": 1,
    "Second round": 2,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Third-place match": 5,
    "Final stage": 5,
    "Final": 5,
}


def _apply_name_map(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team"] = df["home_team"].replace(NAME_MAP)
    df["away_team"] = df["away_team"].replace(NAME_MAP)
    return df


def load_and_clean_matches(path: str = "data/raw/matches_1930_2022.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _apply_name_map(df)

    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]
    df["target"] = np.select(conditions, [0, 1, 2], default=1)

    df["stage"] = df["Round"].map(STAGE_MAP).fillna(1).astype(int)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    teams = set(df["home_team"]) | set(df["away_team"])
    rows = []
    for team in teams:
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]
        total_matches = len(home) + len(away)

        tournaments = set(home["Year"].unique()) | set(away["Year"].unique())
        total_wc_appearances = len(tournaments)

        wins = ((home["target"] == 0).sum()) + ((away["target"] == 2).sum())
        overall_win_rate = wins / total_matches if total_matches > 0 else 0.0

        recent = df[df["Year"] >= 2010]
        rh = recent[recent["home_team"] == team]
        ra = recent[recent["away_team"] == team]
        recent_total = len(rh) + len(ra)
        recent_wins = ((rh["target"] == 0).sum()) + ((ra["target"] == 2).sum())
        recent_win_rate = recent_wins / recent_total if recent_total > 0 else 0.0

        finals = home[home["Round"] == "Final"]
        finals_away = away[away["Round"] == "Final"]
        final_appearances = len(finals) + len(finals_away)

        champion_count = (
            (finals["target"] == 0).sum() + (finals_away["target"] == 2).sum()
        )

        rows.append(
            {
                "team": team,
                "total_wc_appearances": total_wc_appearances,
                "overall_win_rate": overall_win_rate,
                "recent_win_rate": recent_win_rate,
                "final_appearances": final_appearances,
                "champion_count": champion_count,
            }
        )
    return pd.DataFrame(rows).set_index("team")


def build_h2h_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = set()
    for _, row in df.iterrows():
        pair = tuple(sorted([row["home_team"], row["away_team"]]))
        pairs.add(pair)

    for t1, t2 in pairs:
        subset = df[
            ((df["home_team"] == t1) & (df["away_team"] == t2))
            | ((df["home_team"] == t2) & (df["away_team"] == t1))
        ]
        h2h_t1_wins = (
            (subset[subset["home_team"] == t1]["target"] == 0).sum()
            + (subset[subset["away_team"] == t1]["target"] == 2).sum()
        )
        h2h_t2_wins = (
            (subset[subset["home_team"] == t2]["target"] == 0).sum()
            + (subset[subset["away_team"] == t2]["target"] == 2).sum()
        )
        h2h_draws = (subset["target"] == 1).sum()
        rows.append(
            {
                "team_a": t1,
                "team_b": t2,
                "h2h_a_wins": int(h2h_t1_wins),
                "h2h_draws": int(h2h_draws),
                "h2h_b_wins": int(h2h_t2_wins),
            }
        )
    return pd.DataFrame(rows)


def _get_h2h(h2h_df: pd.DataFrame, home: str, away: str):
    pair = tuple(sorted([home, away]))
    match = h2h_df[(h2h_df["team_a"] == pair[0]) & (h2h_df["team_b"] == pair[1])]
    if match.empty:
        return 0, 0, 0
    row = match.iloc[0]
    if pair[0] == home:
        return int(row["h2h_a_wins"]), int(row["h2h_draws"]), int(row["h2h_b_wins"])
    else:
        return int(row["h2h_b_wins"]), int(row["h2h_draws"]), int(row["h2h_a_wins"])


def build_feature_matrix(
    df: pd.DataFrame,
    team_stats: pd.DataFrame,
    h2h_stats: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    ranking = ranking_df.set_index("team")[["rank", "points"]].copy()

    rows = []
    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        home_rank = ranking.loc[home, "rank"] if home in ranking.index else 100.0
        away_rank = ranking.loc[away, "rank"] if away in ranking.index else 100.0
        home_pts = ranking.loc[home, "points"] if home in ranking.index else 0.0
        away_pts = ranking.loc[away, "points"] if away in ranking.index else 0.0

        home_stats = team_stats.loc[home] if home in team_stats.index else pd.Series(dtype=float)
        away_stats = team_stats.loc[away] if away in team_stats.index else pd.Series(dtype=float)

        h2h_home_wins, h2h_draws, h2h_away_wins = _get_h2h(h2h_stats, home, away)

        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "Year": match["Year"],
                "rank_diff": home_rank - away_rank,
                "points_diff": home_pts - away_pts,
                "home_win_rate": home_stats.get("overall_win_rate", 0.0),
                "away_win_rate": away_stats.get("overall_win_rate", 0.0),
                "home_recent_win_rate": home_stats.get("recent_win_rate", 0.0),
                "away_recent_win_rate": away_stats.get("recent_win_rate", 0.0),
                "h2h_home_wins": h2h_home_wins,
                "h2h_draws": h2h_draws,
                "h2h_away_wins": h2h_away_wins,
                "stage": match["stage"],
                "home_appearances": home_stats.get("total_wc_appearances", 0),
                "away_appearances": away_stats.get("total_wc_appearances", 0),
                "home_champion_count": home_stats.get("champion_count", 0),
                "away_champion_count": away_stats.get("champion_count", 0),
                "home_final_appearances": home_stats.get("final_appearances", 0),
                "away_final_appearances": away_stats.get("final_appearances", 0),
                "target": match["target"],
            }
        )

    features_df = pd.DataFrame(rows)
    features_df = features_df.fillna(0)
    return features_df
