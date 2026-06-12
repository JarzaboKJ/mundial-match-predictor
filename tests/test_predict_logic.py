import numpy as np
import pytest

from fetch_current_data import (
    compute_implied_probabilities,
    build_match_record,
    _clamp_int,
    _norm_morale,
)
from predict_today import (
    apply_injury_adjustment,
    apply_form_adjustment,
    apply_morale_adjustment,
    parse_morale,
    safe_form_score,
    valid_bookie_proba,
)


class TestImpliedProbabilities:
    def test_margin_removed(self):
        probs = compute_implied_probabilities(
            [{"home_win": 2.0, "draw": 4.0, "away_win": 4.0}]
        )
        assert probs == pytest.approx((0.5, 0.25, 0.25))
        assert sum(probs) == pytest.approx(1.0)

    def test_empty_returns_none(self):
        assert compute_implied_probabilities([]) is None
        assert compute_implied_probabilities(None) is None

    def test_zero_odds_no_crash(self):
        """Odds of 0 used to raise ZeroDivisionError; null odds, KeyError."""
        assert compute_implied_probabilities(
            [{"home_win": 0, "draw": 4.0, "away_win": 4.0}]
        ) is None
        assert compute_implied_probabilities(
            [{"home_win": None, "draw": 4.0, "away_win": 4.0}]
        ) is None
        assert compute_implied_probabilities([{"home_win": 2.0}]) is None

    def test_invalid_entries_skipped_valid_kept(self):
        probs = compute_implied_probabilities([
            {"home_win": 0, "draw": 0, "away_win": 0},
            {"home_win": 2.0, "draw": 4.0, "away_win": 4.0},
        ])
        assert probs == pytest.approx((0.5, 0.25, 0.25))


class TestSanitization:
    def test_clamp_int(self):
        assert _clamp_int(None, -2, 2) == 0
        assert _clamp_int("3", -2, 2) == 2
        assert _clamp_int(1.6, -2, 2) == 2
        assert _clamp_int(-7, -2, 2) == -2
        assert _clamp_int("abc", 1, 10, default=5) == 5

    def test_norm_morale(self):
        assert _norm_morale("HIGH confidence after win") == "high"
        assert _norm_morale(None) == "medium"
        assert _norm_morale("") == "medium"
        assert _norm_morale("ecstatic") == "medium"

    def test_build_match_record_garbage_llm_output(self):
        """A fully malformed LLM payload must still produce a usable record."""
        record = build_match_record(
            {
                "home_team": "Wrong Name From LLM",
                "home_injuries": [{"importance": "high"}, "not a dict"],
                "home_form_score": "excellent",
                "away_form_score": 5,
                "home_morale": None,
                "bookmaker_odds": [{"home_win": 0, "draw": None, "away_win": 3}],
                "key_news": [123],
            },
            home="Canada",
            away="Bosnia and Herzegovina",
        )
        assert record["home_team"] == "Canada"  # schedule name wins
        assert record["has_bookie_odds"] is False
        assert record["home_form_score"] == 0
        assert record["away_form_score"] == 2  # clamped from 5
        assert record["home_morale"] == "medium"
        assert record["home_injuries"][0]["importance"] == 5  # default
        assert record["key_news"] == ["123"]


class TestValidBookieProba:
    def test_legacy_zero_probs_rejected(self):
        cur = {"bookie_home_prob": 0.0, "bookie_draw_prob": 0.0, "bookie_away_prob": 0.0}
        assert valid_bookie_proba(cur) is None

    def test_explicit_flag_rejected(self):
        cur = {
            "has_bookie_odds": False,
            "bookie_home_prob": 0.0,
            "bookie_draw_prob": 0.0,
            "bookie_away_prob": 0.0,
        }
        assert valid_bookie_proba(cur) is None

    def test_none_record(self):
        assert valid_bookie_proba(None) is None

    def test_null_fields(self):
        cur = {"bookie_home_prob": None, "bookie_draw_prob": None, "bookie_away_prob": None}
        assert valid_bookie_proba(cur) is None

    def test_valid_normalized(self):
        cur = {"bookie_home_prob": 0.5, "bookie_draw_prob": 0.3, "bookie_away_prob": 0.2}
        p = valid_bookie_proba(cur)
        assert p is not None
        assert p.sum() == pytest.approx(1.0)


class TestInjuryAdjustment:
    def test_no_injuries_no_change(self):
        p = np.array([0.5, 0.3, 0.2])
        out, hi, ai = apply_injury_adjustment(p, [], [])
        assert out == pytest.approx(p)
        assert hi == 0 and ai == 0

    def test_underdog_still_penalized(self):
        """Old gate (p > impact) skipped underdogs entirely."""
        p = np.array([0.10, 0.30, 0.60])
        injuries = [{"importance": 10}, {"importance": 8}]  # impact 0.144
        out, hi, _ = apply_injury_adjustment(p, injuries, [])
        assert hi == pytest.approx(0.144)
        assert out[0] < 0.10  # the penalty is actually applied

    def test_impact_capped(self):
        injuries = [{"importance": 10}] * 10
        _, hi, _ = apply_injury_adjustment(np.array([0.5, 0.3, 0.2]), injuries, [])
        assert hi == 0.15

    def test_output_is_distribution(self):
        p = np.array([0.5, 0.3, 0.2])
        out, _, _ = apply_injury_adjustment(
            p, [{"importance": 9}], [{"importance": 7}]
        )
        assert out.sum() == pytest.approx(1.0)
        assert (out > 0).all()


class TestFormAndMorale:
    def test_parse_morale_edge_cases(self):
        assert parse_morale(None) == 0.0
        assert parse_morale("") == 0.0
        assert parse_morale("HIGH confidence") == 0.03
        assert parse_morale("low after defeat") == -0.03
        assert parse_morale("unknown") == 0.0

    def test_safe_form_score(self):
        assert safe_form_score(None) == 0
        assert safe_form_score("2") == 2
        assert safe_form_score(3.7) == 2
        assert safe_form_score(-5) == -2
        assert safe_form_score("great") == 0

    def test_form_adjustment_directions(self):
        p = np.array([0.4, 0.3, 0.3])
        out, adj = apply_form_adjustment(p, 2, -2)
        assert adj == pytest.approx(0.08)
        assert out[0] > 0.4 and out[2] < 0.3

    def test_morale_adjustment(self):
        p = np.array([0.4, 0.3, 0.3])
        out, h, a = apply_morale_adjustment(p, "high", "low")
        assert h == 0.03 and a == -0.03
        assert out.sum() == pytest.approx(1.0)
