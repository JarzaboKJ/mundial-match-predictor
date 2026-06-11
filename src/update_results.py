import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Update match results in results.json")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument(
        "--result",
        required=True,
        choices=["home_win", "draw", "away_win"],
        help="Actual match result",
    )
    args = parser.parse_args()

    path = "results/results.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run predict_today.py first.")
        sys.exit(1)

    found = False
    for match in data["matches"]:
        if match["home_team"] == args.home and match["away_team"] == args.away:
            match["actual_result"] = args.result
            match["correct"] = match["predicted_winner"] == args.result
            found = True
            status = "TRAFIONY" if match["correct"] else "PUDLO"
            print(f"{args.home} vs {args.away}: wynik={args.result}, predykcja={match['predicted_winner']} -> {status}")
            break

    if not found:
        print(f"ERROR: Match {args.home} vs {args.away} not found in results.json")
        sys.exit(1)

    evaluated = [m for m in data["matches"] if m["actual_result"] is not None]
    total = len(evaluated)
    correct = sum(1 for m in evaluated if m["correct"])
    accuracy = correct / total if total > 0 else 0.0

    data["stats"] = {
        "total_predictions": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Accuracy po {total} meczach: {accuracy*100:.1f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
