import pandas as pd
import pytest

from preprocessing import (
    FEATURE_COLS,
    canonical_name,
    clean_matches,
    build_feature_matrix,
    build_team_stats,
    build_h2h_stats,
    _ranking_fallbacks,
    _prep_ranking,
)


def make_matches(rows):
    df = pd.DataFrame(
        rows,
        columns=["home_team", "away_team", "home_score", "away_score", "Round", "Date", "Year"],
    )
    return clean_matches(df)


RANKING = pd.DataFrame(
    {
        "team": ["Brazil", "Germany", "Czechia", "Italy"],
        "rank": [1, 2, 30, 8],
        "points": [1840.0, 1820.0, 1500.0, 1700.0],
    }
)

WORLD_CUP = pd.DataFrame(
    {
        "Year": [1994, 2002, 2014],
        "Champion": ["Brazil", "Brazil", "Germany"],
        "Runner-Up": ["Italy", "Germany", "Argentina"],
    }
)


class TestCanonicalNames:
    def test_successor_states(self):
        assert canonical_name("Czechoslovakia") == "Czechia"
        assert canonical_name("Czech Republic") == "Czechia"
        assert canonical_name("West Germany") == "Germany"
        assert canonical_name("Zaire") == "Congo DR"
        assert canonical_name("United States") == "USA"

    def test_unmapped_passthrough(self):
        assert canonical_name("Brazil") == "Brazil"

    def test_clean_matches_merges_predecessors(self):
        df = make_matches([
            ["West Germany", "Czechoslovakia", 2, 1, "Final", "1990-07-08", 1990],
        ])
        assert df.iloc[0]["home_team"] == "Germany"
        assert df.iloc[0]["away_team"] == "Czechia"


class TestTargets:
    def test_targets(self):
        df = make_matches([
            ["Brazil", "Italy", 2, 0, "Group stage", "1994-06-01", 1994],
            ["Brazil", "Italy", 1, 1, "Group stage", "1994-06-05", 1994],
            ["Brazil", "Italy", 0, 3, "Group stage", "1994-06-09", 1994],
        ])
        assert list(df["target"]) == [0, 1, 2]

    def test_round_of_32_mapped(self):
        df = make_matches([
            ["Brazil", "Italy", 2, 0, "Round of 32", "2026-06-29", 2026],
        ])
        assert df.iloc[0]["stage"] == 2


class TestLeakageFreeFeatures:
    def test_first_meeting_has_zero_h2h(self):
        """A pair's first match must not see its own outcome in features."""
        df = make_matches([
            ["Brazil", "Italy", 1, 1, "Group stage", "1994-06-01", 1994],
        ])
        feats = build_feature_matrix(df, WORLD_CUP, RANKING)
        row = feats.iloc[0]
        assert row["h2h_home_wins"] == 0
        assert row["h2h_draws"] == 0
        assert row["h2h_away_wins"] == 0
        assert row["home_win_rate"] == 0.0
        assert row["away_win_rate"] == 0.0

    def test_second_meeting_sees_only_first(self):
        df = make_matches([
            ["Brazil", "Italy", 1, 1, "Group stage", "1994-06-01", 1994],
            ["Italy", "Brazil", 0, 2, "Final", "1994-07-17", 1994],
            ["Brazil", "Italy", 0, 1, "Group stage", "2002-06-01", 2002],
        ])
        feats = build_feature_matrix(df, WORLD_CUP, RANKING)
        final_row = feats.iloc[1]
        assert final_row["h2h_draws"] == 1
        assert final_row["h2h_home_wins"] == 0  # Italy at home in the final
        third = feats.iloc[2]
        assert third["h2h_home_wins"] == 1  # Brazil's win in the 1994 final
        assert third["h2h_draws"] == 1

    def test_champion_count_is_pre_tournament(self):
        df = make_matches([
            ["Brazil", "Italy", 2, 0, "Group stage", "1994-06-01", 1994],
            ["Brazil", "Germany", 2, 0, "Final", "2002-06-30", 2002],
        ])
        feats = build_feature_matrix(df, WORLD_CUP, RANKING)
        # In 1994 Brazil had no titles yet (in this fixture); in 2002 one (1994).
        assert feats.iloc[0]["home_champion_count"] == 0
        assert feats.iloc[1]["home_champion_count"] == 1
        # Germany was runner-up in 2002 — that final appearance must not be
        # visible to the 2002 final itself.
        assert feats.iloc[1]["away_final_appearances"] == 0

    def test_appearances_strictly_before(self):
        df = make_matches([
            ["Brazil", "Italy", 2, 0, "Group stage", "1994-06-01", 1994],
            ["Brazil", "Italy", 2, 0, "Group stage", "2002-06-01", 2002],
        ])
        feats = build_feature_matrix(df, WORLD_CUP, RANKING)
        assert feats.iloc[0]["home_appearances"] == 0
        assert feats.iloc[1]["home_appearances"] == 1

    def test_feature_columns_complete(self):
        df = make_matches([
            ["Brazil", "Italy", 2, 0, "Group stage", "1994-06-01", 1994],
        ])
        feats = build_feature_matrix(df, WORLD_CUP, RANKING)
        for col in FEATURE_COLS:
            assert col in feats.columns


class TestTeamStats:
    def test_champion_count_includes_penalty_finals(self):
        """Champions come from world_cup.csv, so a 0-0 final (decided on
        penalties) still produces a champion."""
        df = make_matches([
            ["Brazil", "Italy", 0, 0, "Final", "1994-07-17", 1994],
        ])
        stats = build_team_stats(df, WORLD_CUP, as_of_year=2026)
        assert stats.loc["Brazil", "champion_count"] == 2  # 1994 + 2002
        assert stats.loc["Brazil", "final_appearances"] == 2
        assert stats.loc["Italy", "champion_count"] == 0
        assert stats.loc["Italy", "final_appearances"] == 1

    def test_as_of_year_excludes_future(self):
        df = make_matches([
            ["Brazil", "Italy", 2, 0, "Group stage", "1994-06-01", 1994],
        ])
        stats = build_team_stats(df, WORLD_CUP, as_of_year=2000)
        assert stats.loc["Brazil", "champion_count"] == 1  # 2002 not yet won


class TestRankingFallbacks:
    def test_unranked_team_worse_than_worst(self):
        ranking = _prep_ranking(RANKING)
        rank_fb, pts_fb = _ranking_fallbacks(ranking)
        assert rank_fb > RANKING["rank"].max()
        assert pts_fb == RANKING["points"].min()


class TestH2HStats:
    def test_orientation(self):
        df = make_matches([
            ["Brazil", "Italy", 2, 0, "Group stage", "1994-06-01", 1994],
            ["Italy", "Brazil", 1, 0, "Group stage", "2002-06-01", 2002],
        ])
        h2h = build_h2h_stats(df)
        row = h2h.iloc[0]
        # Pair is alphabetically sorted: team_a=Brazil, team_b=Italy.
        assert row["team_a"] == "Brazil"
        assert row["h2h_a_wins"] == 1
        assert row["h2h_b_wins"] == 1
        assert row["h2h_draws"] == 0
