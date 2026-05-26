import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "datasets" / "test.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "baseline" / "baseline_outputs.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_ids(path):
    if not path.exists():
        return set()
    return {record["id"] for record in read_jsonl(path) if "id" in record}


def load_model(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


def generate_text(model, tokenizer, messages, max_new_tokens=512, temperature=0.4, top_p=0.9):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def build_prompt(record):
    return f"""You are reviewing a student's answer in a software engineering interview.

Question:
{record["question"]}

Reference answer:
{record["reference_answer"]}

Student answer:
{record["student_answer"]}

Write coach feedback for the student. The feedback must:
- Identify what the student got right.
- Identify missing or incorrect technical details.
- Give specific, actionable improvement advice.
- Stay concise and professional.
- Avoid giving a full rewritten answer unless needed.
"""


def generate_feedback(model, tokenizer, record):
    messages = [
        {
            "role": "system",
            "content": "You are a precise and helpful AI interview coach for CS students.",
        },
        {"role": "user", "content": build_prompt(record)},
    ]
    return generate_text(model, tokenizer, messages).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate baseline coach feedback for the held-out test set.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output and regenerate.")
    args = parser.parse_args()

    if args.overwrite and args.output.exists():
        args.output.unlink()

    records = read_jsonl(args.input)
    existing_ids = load_existing_ids(args.output)
    model, tokenizer = load_model(args.model)

    generated_count = 0
    for index, record in enumerate(records):
        if args.limit is not None and generated_count >= args.limit:
            break
        if record.get("id") in existing_ids:
            continue
        if not record.get("student_answer"):
            raise ValueError(f"Missing student_answer for {record.get('id')}. Run generate_test_student_answers.py first.")

        print(f"[{index + 1}/{len(records)}] Generating baseline feedback for {record.get('id')}")
        feedback = generate_feedback(model, tokenizer, record)

        output_record = {
            **record,
            "baseline_feedback": feedback,
            "baseline_model": args.model,
        }
        append_jsonl(output_record, args.output)
        existing_ids.add(record["id"])
        generated_count += 1

    print(f"Generated {generated_count} baseline feedback records.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
