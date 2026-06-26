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
    parser.add_argument(
        "--date",
        default=None,
        help="Match date (ISO). Required when the same teams met more than once.",
    )
    parser.add_argument(
        "--no-prediction",
        action="store_true",
        help="Log result without a prediction (training data only, not counted in accuracy).",
    )
    args = parser.parse_args()

    path = "results/results.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Run predict_today.py first.")
        sys.exit(1)

    candidates = [
        m for m in data["matches"]
        if m["home_team"] == args.home and m["away_team"] == args.away
        and (args.date is None or m["date"] == args.date)
    ]

    if args.no_prediction:
        if len(candidates) > 1:
            print(f"ERROR: {len(candidates)} matches found for {args.home} vs {args.away}. "
                  f"Disambiguate with --date:")
            for m in candidates:
                print(f"  --date {m['date']} (actual: {m['actual_result']})")
            sys.exit(1)

        if candidates:
            match = candidates[0]
            match["actual_result"] = args.result
            match["predicted_winner"] = None
            match["model_proba"] = None
            match["final_proba"] = None
            match["correct"] = None
            print(f"Updated: {args.home} vs {args.away} ({match['date']}): "
                  f"wynik={args.result} [bez predykcji, tylko dane treningowe]")
        else:
            if args.date is None:
                print("ERROR: --date is required when adding a new match with --no-prediction.")
                sys.exit(1)
            data["matches"].append({
                "date": args.date,
                "home_team": args.home,
                "away_team": args.away,
                "model_proba": None,
                "final_proba": None,
                "predicted_winner": None,
                "actual_result": args.result,
                "correct": None,
            })
            print(f"Added: {args.home} vs {args.away} ({args.date}): "
                  f"wynik={args.result} [bez predykcji, tylko dane treningowe]")

    else:
        if not candidates:
            print(f"ERROR: Match {args.home} vs {args.away}"
                  f"{' on ' + args.date if args.date else ''} not found in results.json")
            sys.exit(1)

        if len(candidates) > 1:
            print(f"ERROR: {len(candidates)} matches found for {args.home} vs {args.away}. "
                  f"Disambiguate with --date:")
            for m in candidates:
                print(f"  --date {m['date']} (predicted: {m['predicted_winner']}, "
                      f"actual: {m['actual_result']})")
            sys.exit(1)

        match = candidates[0]
        if match["actual_result"] is not None and match["actual_result"] != args.result:
            print(f"NOTE: overwriting previously recorded result "
                  f"{match['actual_result']} -> {args.result}")

        match["actual_result"] = args.result
        match["correct"] = match["predicted_winner"] == args.result
        status = "TRAFIONY" if match["correct"] else "PUDLO"
        print(f"{args.home} vs {args.away} ({match['date']}): wynik={args.result}, "
              f"predykcja={match['predicted_winner']} -> {status}")

    # Accuracy counts only matches that had a prediction (correct is not None)
    evaluated = [m for m in data["matches"] if m.get("correct") is not None]
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

    print(f"Accuracy po {total} predykcjach: {accuracy*100:.1f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
