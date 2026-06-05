import argparse
import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_BASELINE_PATH = ROOT_DIR / "baseline" / "baseline_scores.jsonl"
DEFAULT_NEW_MODEL_PATH = ROOT_DIR / "newModel" / "new_model_scores.jsonl"
DEFAULT_SUMMARY_PATH = ROOT_DIR / "eval" / "comparison_summary.json"
DEFAULT_TABLE_PATH = ROOT_DIR / "eval" / "comparison_table.csv"
DEFAULT_EXAMPLES_PATH = ROOT_DIR / "eval" / "comparison_examples.csv"

SCORE_FIELDS = [
    "technical_correctness",
    "specificity",
    "helpfulness",
    "actionability",
    "interview_coaching_quality",
    "overall_score",
]


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    return records


def records_by_id(records, source_name):
    by_id = {}
    duplicates = []
    for record in records:
        record_id = record.get("id")
        if not record_id:
            raise ValueError(f"{source_name} contains a record without an id")
        if record_id in by_id:
            duplicates.append(record_id)
        by_id[record_id] = record
    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"{source_name} contains duplicate ids: {duplicate_list}")
    return by_id


def average(records, field):
    values = [float(record[field]) for record in records if field in record]
    if not values:
        raise ValueError(f"No values found for field: {field}")
    return sum(values) / len(values)


def round_score(value):
    return round(value, 4)


def build_comparison(baseline_records, new_model_records):
    baseline_by_id = records_by_id(baseline_records, "baseline scores")
    new_by_id = records_by_id(new_model_records, "new model scores")

    baseline_ids = set(baseline_by_id)
    new_ids = set(new_by_id)
    matched_ids = sorted(baseline_ids & new_ids)
    missing_in_new = sorted(baseline_ids - new_ids)
    missing_in_baseline = sorted(new_ids - baseline_ids)

    if not matched_ids:
        raise ValueError("No matching ids found between baseline and new model scores")

    matched_baseline = [baseline_by_id[record_id] for record_id in matched_ids]
    matched_new = [new_by_id[record_id] for record_id in matched_ids]

    dimension_rows = []
    for field in SCORE_FIELDS:
        baseline_avg = average(matched_baseline, field)
        new_avg = average(matched_new, field)
        dimension_rows.append(
            {
                "metric": field,
                "baseline_average": round_score(baseline_avg),
                "new_model_average": round_score(new_avg),
                "delta": round_score(new_avg - baseline_avg),
            }
        )

    example_rows = []
    new_wins = 0
    baseline_wins = 0
    ties = 0

    for record_id in matched_ids:
        baseline_score = float(baseline_by_id[record_id]["overall_score"])
        new_score = float(new_by_id[record_id]["overall_score"])
        delta = new_score - baseline_score

        if delta > 0:
            winner = "new_model"
            new_wins += 1
        elif delta < 0:
            winner = "baseline"
            baseline_wins += 1
        else:
            winner = "tie"
            ties += 1

        example_rows.append(
            {
                "id": record_id,
                "baseline_overall_score": round_score(baseline_score),
                "new_model_overall_score": round_score(new_score),
                "delta": round_score(delta),
                "winner": winner,
            }
        )

    total = len(matched_ids)
    non_tie_total = new_wins + baseline_wins
    summary = {
        "total_matched_examples": total,
        "baseline_records": len(baseline_records),
        "new_model_records": len(new_model_records),
        "missing_in_new_model": missing_in_new,
        "missing_in_baseline": missing_in_baseline,
        "win_loss_tie": {
            "new_model_wins": new_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "new_model_win_rate_all": round_score(new_wins / total),
            "new_model_win_rate_excluding_ties": round_score(new_wins / non_tie_total)
            if non_tie_total
            else None,
        },
        "dimension_averages": dimension_rows,
    }

    example_rows.sort(key=lambda row: row["delta"], reverse=True)
    return summary, dimension_rows, example_rows


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and new model feedback scores.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--new-model", type=Path, default=DEFAULT_NEW_MODEL_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--examples-output", type=Path, default=DEFAULT_EXAMPLES_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_records = read_jsonl(args.baseline)
    new_model_records = read_jsonl(args.new_model)

    summary, dimension_rows, example_rows = build_comparison(baseline_records, new_model_records)

    write_json(args.summary_output, summary)
    write_csv(
        args.table_output,
        dimension_rows,
        ["metric", "baseline_average", "new_model_average", "delta"],
    )
    write_csv(
        args.examples_output,
        example_rows,
        ["id", "baseline_overall_score", "new_model_overall_score", "delta", "winner"],
    )

    overall = next(row for row in dimension_rows if row["metric"] == "overall_score")
    wins = summary["win_loss_tie"]
    print(f"Matched examples: {summary['total_matched_examples']}")
    print(
        "Overall score: "
        f"baseline={overall['baseline_average']}, "
        f"new_model={overall['new_model_average']}, "
        f"delta={overall['delta']:+.4f}"
    )
    print(
        "Win/loss/tie: "
        f"new_model={wins['new_model_wins']}, "
        f"baseline={wins['baseline_wins']}, "
        f"ties={wins['ties']}"
    )
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.table_output}")
    print(f"Wrote {args.examples_output}")


if __name__ == "__main__":
    main()
