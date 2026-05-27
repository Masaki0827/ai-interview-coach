import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "preference_candidates" / "preference_candidates.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "preference_candidates" / "preference_pairs.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"


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
    print(f"Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_text(model, tokenizer, messages, max_new_tokens=512, temperature=0.1, top_p=0.9):
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


def extract_json_object(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text[:300]}")
    return json.loads(match.group(0))


def build_prompt(record, feedback_a_field, feedback_b_field):
    return f"""Choose the better coach feedback for a software engineering interview answer.

Question:
{record.get("question", "")}

Reference answer:
{record.get("reference_answer", "")}

Student answer:
{record.get("student_answer", "")}

Feedback A:
{record.get(feedback_a_field, "")}

Feedback B:
{record.get(feedback_b_field, "")}

Select the feedback that is more technically correct, specific, helpful, actionable, and suitable for interview coaching.
If both are imperfect, choose the one that would better help the student improve.

Return only valid JSON in this exact format:
{{
  "winner": "a",
  "reason": "..."
}}
"""


def choose_feedback(model, tokenizer, record, feedback_a_field, feedback_b_field):
    messages = [
        {
            "role": "system",
            "content": "You are a strict, consistent evaluator of interview coaching feedback preferences.",
        },
        {"role": "user", "content": build_prompt(record, feedback_a_field, feedback_b_field)},
    ]
    raw_output = generate_text(model, tokenizer, messages)
    parsed = extract_json_object(raw_output)
    winner = str(parsed.get("winner", "")).strip().lower()
    if winner not in {"a", "b"}:
        raise ValueError(f"Invalid winner from judge: {winner}")

    chosen_field = feedback_a_field if winner == "a" else feedback_b_field
    rejected_field = feedback_b_field if winner == "a" else feedback_a_field

    return {
        "id": record.get("id"),
        "question": record.get("question", ""),
        "reference_answer": record.get("reference_answer", ""),
        "student_answer": record.get("student_answer", ""),
        "student_answer_type": record.get("student_answer_type", ""),
        "chosen_feedback": record.get(chosen_field, ""),
        "rejected_feedback": record.get(rejected_field, ""),
        "winner": winner,
        "reason": str(parsed.get("reason", "")).strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Choose preferred feedback with a judge model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--feedback-a-field", default="feedback_a")
    parser.add_argument("--feedback-b-field", default="feedback_b")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite and args.output.exists():
        args.output.unlink()

    records = read_jsonl(args.input)
    existing_ids = load_existing_ids(args.output)
    model, tokenizer = load_model(args.model)

    scored_count = 0
    for index, record in enumerate(records):
        if args.limit is not None and scored_count >= args.limit:
            break
        if record.get("id") in existing_ids:
            continue
        if not record.get(args.feedback_a_field) or not record.get(args.feedback_b_field):
            raise ValueError(f"Missing feedback candidates for {record.get('id')}")

        print(f"[{index + 1}/{len(records)}] Choosing preference for {record.get('id')}")
        preference_record = choose_feedback(model, tokenizer, record, args.feedback_a_field, args.feedback_b_field)
        preference_record["judge_model"] = args.model
        append_jsonl(preference_record, args.output)
        existing_ids.add(record["id"])
        scored_count += 1

    print(f"Scored {scored_count} preference records.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
