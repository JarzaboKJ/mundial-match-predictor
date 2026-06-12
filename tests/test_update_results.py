import json
import sys

import pytest

import update_results


def write_results(tmp_path, matches):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    payload = {
        "matches": matches,
        "stats": {"total_predictions": 0, "correct": 0, "accuracy": 0.0},
    }
    (results_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    return results_dir / "results.json"


def run_cli(monkeypatch, tmp_path, argv):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["update_results.py"] + argv)
    update_results.main()


BASE_MATCH = {
    "model_proba": {}, "bookie_proba": {}, "final_proba": {},
    "actual_result": None, "correct": None,
}


def test_updates_match_and_stats(tmp_path, monkeypatch):
    path = write_results(tmp_path, [
        {**BASE_MATCH, "date": "2026-06-12", "home_team": "Canada",
         "away_team": "Bosnia and Herzegovina", "predicted_winner": "home_win"},
    ])
    run_cli(monkeypatch, tmp_path,
            ["--home", "Canada", "--away", "Bosnia and Herzegovina", "--result", "home_win"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["matches"][0]["actual_result"] == "home_win"
    assert data["matches"][0]["correct"] is True
    assert data["stats"] == {"total_predictions": 1, "correct": 1, "accuracy": 1.0}


def test_rematch_requires_date(tmp_path, monkeypatch):
    """Same pairing twice (group + knockout): refuse to guess which one."""
    write_results(tmp_path, [
        {**BASE_MATCH, "date": "2026-06-12", "home_team": "Canada",
         "away_team": "Qatar", "predicted_winner": "home_win"},
        {**BASE_MATCH, "date": "2026-07-04", "home_team": "Canada",
         "away_team": "Qatar", "predicted_winner": "draw"},
    ])
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, tmp_path,
                ["--home", "Canada", "--away", "Qatar", "--result", "home_win"])


def test_rematch_with_date_updates_correct_one(tmp_path, monkeypatch):
    path = write_results(tmp_path, [
        {**BASE_MATCH, "date": "2026-06-12", "home_team": "Canada",
         "away_team": "Qatar", "predicted_winner": "home_win"},
        {**BASE_MATCH, "date": "2026-07-04", "home_team": "Canada",
         "away_team": "Qatar", "predicted_winner": "draw"},
    ])
    run_cli(monkeypatch, tmp_path,
            ["--home", "Canada", "--away", "Qatar", "--result", "draw",
             "--date", "2026-07-04"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["matches"][0]["actual_result"] is None
    assert data["matches"][1]["actual_result"] == "draw"
    assert data["matches"][1]["correct"] is True


def test_unknown_match_errors(tmp_path, monkeypatch):
    write_results(tmp_path, [])
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, tmp_path,
                ["--home", "Atlantis", "--away", "Qatar", "--result", "draw"])
