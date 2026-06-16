from bisect import bisect_left
from collections import defaultdict

import pandas as pd
import numpy as np

# Canonical team names = current FIFA ranking names. Historical sources use
# predecessor states and alternate spellings; everything is mapped here so that
# matches, rankings, world_cup.csv and the 2026 schedule join on one name.
NAME_MAP = {
    "United States": "USA",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "IR Iran",
    # Successor states / renames (FIFA attributes the historical record
    # of the predecessor to the current federation):
    "Czech Republic": "Czechia",
    "Czechoslovakia": "Czechia",
    "West Germany": "Germany",
    "Zaire": "Congo DR",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Dutch East Indies": "Indonesia",
}

STAGE_MAP = {
    "Group stage": 1,
    "First group stage": 1,
    "Second group stage": 1,
    "Group stage play-off": 1,
    "First round": 1,
    "Second round": 2,
    "Round of 32": 2,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Third-place match": 5,
    "Final stage": 5,
    "Final": 5,
}

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

# Window (in years) defining "recent" form: the last 4 tournaments.
RECENT_WINDOW_YEARS = 16


def canonical_name(team: str) -> str:
    return NAME_MAP.get(team, team)


def _apply_name_map(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team"] = df["home_team"].replace(NAME_MAP)
    df["away_team"] = df["away_team"].replace(NAME_MAP)
    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
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


def load_and_clean_matches(path: str = "data/raw/matches_1930_2022.csv") -> pd.DataFrame:
    return clean_matches(pd.read_csv(path))


def load_ranking(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["team"] = df["team"].replace(NAME_MAP)
    return df


def load_world_cup(path: str = "data/raw/world_cup.csv") -> pd.DataFrame:
    """Tournament summary: authoritative champion / runner-up per year.

    This is the only reliable source for titles — deriving champions from the
    "Final" round misses penalty-shootout finals (1994, 2006, 2022 ended level
    after 120') and 1950, which had a final round-robin instead of a final.
    """
    df = pd.read_csv(path)
    df = df.copy()
    df["Champion"] = df["Champion"].replace(NAME_MAP)
    df["Runner-Up"] = df["Runner-Up"].replace(NAME_MAP)
    return df


def _title_years(world_cup_df: pd.DataFrame):
    """Per team: sorted years of titles and of final (or decider) appearances."""
    champion_years = defaultdict(list)
    final_years = defaultdict(list)
    for _, row in world_cup_df.iterrows():
        year = int(row["Year"])
        champion_years[row["Champion"]].append(year)
        final_years[row["Champion"]].append(year)
        final_years[row["Runner-Up"]].append(year)
    for d in (champion_years, final_years):
        for team in d:
            d[team].sort()
    return champion_years, final_years


def _ranking_fallbacks(ranking: pd.DataFrame):
    """Imputation for teams absent from a ranking file.

    rank=100/points=0 (the old fallback) put unranked teams mid-table on rank
    but ~1500 points below everyone on points_diff; use a consistent
    'worse than the worst ranked team' for both.
    """
    return float(ranking["rank"].max()) + 10.0, float(ranking["points"].min())


def _prep_ranking(ranking_df: pd.DataFrame) -> pd.DataFrame:
    ranking = ranking_df.copy()
    ranking["team"] = ranking["team"].replace(NAME_MAP)
    return ranking.set_index("team")[["rank", "points"]]


class _TeamAccumulator:
    """Chronological per-team stats so each match only sees its past."""

    def __init__(self):
        self.matches = 0
        self.wins = 0
        self.results = []  # (year, won) per match, in date order
        self.years = set()  # tournaments played strictly before current match


def build_feature_matrix(
    df: pd.DataFrame,
    world_cup_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the training matrix WITHOUT target leakage.

    Every row's features are computed exclusively from matches played strictly
    before that row's date (and titles won in strictly earlier tournaments).
    The previous implementation computed team/h2h stats over the full dataset,
    so each row's own outcome — and the test set — leaked into its features;
    47% of rows (pairs that met exactly once) had the label fully encoded in
    the h2h columns.
    """
    ranking = _prep_ranking(ranking_df)
    rank_fb, pts_fb = _ranking_fallbacks(ranking)
    champion_years, final_years = _title_years(world_cup_df)

    df = df.sort_values("Date", kind="stable").reset_index(drop=True)

    teams = defaultdict(_TeamAccumulator)
    h2h = defaultdict(lambda: [0, 0, 0])  # sorted pair -> [first_wins, draws, second_wins]

    rows = []
    for match in df.itertuples():
        home, away = match.home_team, match.away_team
        year = int(match.Year)

        hs, as_ = teams[home], teams[away]

        def win_rate(acc):
            return acc.wins / acc.matches if acc.matches else 0.0

        def recent_win_rate(acc):
            cutoff = year - RECENT_WINDOW_YEARS
            recent = [(y, w) for y, w in acc.results if y > cutoff]
            return sum(w for _, w in recent) / len(recent) if recent else 0.0

        pair = tuple(sorted([home, away]))
        a_wins, draws, b_wins = h2h[pair]
        if pair[0] == home:
            h2h_hw, h2h_d, h2h_aw = a_wins, draws, b_wins
        else:
            h2h_hw, h2h_d, h2h_aw = b_wins, draws, a_wins

        home_rank = ranking.loc[home, "rank"] if home in ranking.index else rank_fb
        away_rank = ranking.loc[away, "rank"] if away in ranking.index else rank_fb
        home_pts = ranking.loc[home, "points"] if home in ranking.index else pts_fb
        away_pts = ranking.loc[away, "points"] if away in ranking.index else pts_fb

        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "Year": year,
                "rank_diff": home_rank - away_rank,
                "points_diff": home_pts - away_pts,
                "home_win_rate": win_rate(hs),
                "away_win_rate": win_rate(as_),
                "home_recent_win_rate": recent_win_rate(hs),
                "away_recent_win_rate": recent_win_rate(as_),
                "h2h_home_wins": h2h_hw,
                "h2h_draws": h2h_d,
                "h2h_away_wins": h2h_aw,
                "stage": match.stage,
                "home_appearances": len(hs.years),
                "away_appearances": len(as_.years),
                "home_champion_count": bisect_left(champion_years.get(home, []), year),
                "away_champion_count": bisect_left(champion_years.get(away, []), year),
                "home_final_appearances": bisect_left(final_years.get(home, []), year),
                "away_final_appearances": bisect_left(final_years.get(away, []), year),
                "target": match.target,
            }
        )

        # Update accumulators AFTER snapshotting features.
        home_won = match.target == 0
        away_won = match.target == 2
        hs.matches += 1
        as_.matches += 1
        hs.wins += int(home_won)
        as_.wins += int(away_won)
        hs.results.append((year, int(home_won)))
        as_.results.append((year, int(away_won)))
        hs.years.add(year)
        as_.years.add(year)

        if pair[0] == home:
            h2h[pair][0] += int(home_won)
            h2h[pair][2] += int(away_won)
        else:
            h2h[pair][0] += int(away_won)
            h2h[pair][2] += int(home_won)
        h2h[pair][1] += int(match.target == 1)

    return pd.DataFrame(rows)


def build_team_stats(
    df: pd.DataFrame,
    world_cup_df: pd.DataFrame,
    as_of_year: int = 2026,
) -> pd.DataFrame:
    """Per-team stats from all matches strictly before `as_of_year`.

    Used at prediction time; semantics match build_feature_matrix exactly
    (appearances/titles strictly before the tournament, recent = last
    RECENT_WINDOW_YEARS years).
    """
    df = df[df["Year"] < as_of_year]
    champion_years, final_years = _title_years(world_cup_df)
    recent_cutoff = as_of_year - RECENT_WINDOW_YEARS

    teams = set(df["home_team"]) | set(df["away_team"])
    rows = []
    for team in teams:
        home = df[df["home_team"] == team]
        away = df[df["away_team"] == team]
        total_matches = len(home) + len(away)

        tournaments = set(home["Year"].unique()) | set(away["Year"].unique())

        wins = ((home["target"] == 0).sum()) + ((away["target"] == 2).sum())
        overall_win_rate = wins / total_matches if total_matches > 0 else 0.0

        rh = home[home["Year"] > recent_cutoff]
        ra = away[away["Year"] > recent_cutoff]
        recent_total = len(rh) + len(ra)
        recent_wins = ((rh["target"] == 0).sum()) + ((ra["target"] == 2).sum())
        recent_win_rate = recent_wins / recent_total if recent_total > 0 else 0.0

        rows.append(
            {
                "team": team,
                "total_wc_appearances": len(tournaments),
                "overall_win_rate": overall_win_rate,
                "recent_win_rate": recent_win_rate,
                "final_appearances": bisect_left(final_years.get(team, []), as_of_year),
                "champion_count": bisect_left(champion_years.get(team, []), as_of_year),
            }
        )
    return pd.DataFrame(rows).set_index("team")


def build_h2h_stats(df: pd.DataFrame, as_of_year: int = 2026) -> pd.DataFrame:
    df = df[df["Year"] < as_of_year]
    pairs = set()
    for _, row in df.iterrows():
        pairs.add(tuple(sorted([row["home_team"], row["away_team"]])))

    rows = []
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


def build_single_match_features(
    home_team: str,
    away_team: str,
    stage: int,
    team_stats: pd.DataFrame,
    h2h_stats: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    home = canonical_name(home_team)
    away = canonical_name(away_team)

    ranking = _prep_ranking(ranking_df)
    rank_fb, pts_fb = _ranking_fallbacks(ranking)

    home_rank = ranking.loc[home, "rank"] if home in ranking.index else rank_fb
    away_rank = ranking.loc[away, "rank"] if away in ranking.index else rank_fb
    home_pts = ranking.loc[home, "points"] if home in ranking.index else pts_fb
    away_pts = ranking.loc[away, "points"] if away in ranking.index else pts_fb

    hs = team_stats.loc[home] if home in team_stats.index else pd.Series(dtype=float)
    as_ = team_stats.loc[away] if away in team_stats.index else pd.Series(dtype=float)

    h2h_hw, h2h_d, h2h_aw = _get_h2h(h2h_stats, home, away)

    row = {
        "rank_diff": home_rank - away_rank,
        "points_diff": home_pts - away_pts,
        "home_win_rate": hs.get("overall_win_rate", 0.0),
        "away_win_rate": as_.get("overall_win_rate", 0.0),
        "home_recent_win_rate": hs.get("recent_win_rate", 0.0),
        "away_recent_win_rate": as_.get("recent_win_rate", 0.0),
        "h2h_home_wins": h2h_hw,
        "h2h_draws": h2h_d,
        "h2h_away_wins": h2h_aw,
        "stage": stage,
        "home_appearances": hs.get("total_wc_appearances", 0),
        "away_appearances": as_.get("total_wc_appearances", 0),
        "home_champion_count": hs.get("champion_count", 0),
        "away_champion_count": as_.get("champion_count", 0),
        "home_final_appearances": hs.get("final_appearances", 0),
        "away_final_appearances": as_.get("final_appearances", 0),
    }

    df = pd.DataFrame([row], columns=FEATURE_COLS).fillna(0)
    return df
