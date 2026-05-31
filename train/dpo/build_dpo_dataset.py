import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT_DIR / "train" / "preference_pairs.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "train" / "dpo" / "dpo_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are a precise and helpful AI interview coach for CS students. "
    "Give feedback directly to the student using second person. "
    "Identify what is correct, what is missing or incorrect, and how to improve the answer."
)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_user_prompt(record):
    return f"""Question:
{record.get("question", "")}

Reference answer:
{record.get("reference_answer", "")}

Student answer:
{record.get("student_answer", "")}

Write interview coaching feedback for the student."""


def convert_record(record):
    chosen_feedback = str(record.get("chosen_feedback", "")).strip()
    rejected_feedback = str(record.get("rejected_feedback", "")).strip()
    if not chosen_feedback:
        raise ValueError(f"Missing chosen_feedback for {record.get('id')}")
    if not rejected_feedback:
        raise ValueError(f"Missing rejected_feedback for {record.get('id')}")

    return {
        "id": record.get("id"),
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ],
        "chosen": [{"role": "assistant", "content": chosen_feedback}],
        "rejected": [{"role": "assistant", "content": rejected_feedback}],
        "source": "preference_pairs",
        "judge_model": record.get("judge_model", ""),
        "winner": record.get("winner", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a DPO dataset from preference pairs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    converted = [convert_record(record) for record in records]
    write_jsonl(converted, args.output)

    print(f"Read: {len(records)} records from {args.input}")
    print(f"Wrote: {len(converted)} DPO records to {args.output}")


if __name__ == "__main__":
    main()
