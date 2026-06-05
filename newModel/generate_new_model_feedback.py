import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "test.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "newModel" / "new_model_outputs.jsonl"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER_PATH = ROOT_DIR / "train" / "model" / "dpo_adapter"
DEFAULT_FEEDBACK_FIELD = "new_model_feedback"


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8-sig") as f:
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


def load_model(base_model_name, adapter_path):
    print(f"Loading base model: {base_model_name}")
    print(f"Loading adapter: {adapter_path}")

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, messages, max_new_tokens=512, temperature=0.4, top_p=0.9):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
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
    return f"""You are writing coach feedback directly to a software engineering interview candidate.

Question:
{record.get("question", "")}

Reference answer:
{record.get("reference_answer", "")}

Student answer:
{record.get("student_answer", "")}

Write interview coaching feedback for the student.

Requirements:
- Address the student directly using "you" and "your answer".
- Identify what the student got right.
- Identify missing, vague, or incorrect technical details.
- Give concrete, actionable improvement advice.
- Keep a professional coaching tone.
- Do not simply rewrite the reference answer.
- Keep the feedback concise.
"""


def generate_feedback(model, tokenizer, record, max_new_tokens, temperature, top_p):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise and helpful AI interview coach for CS students. "
                "Always speak directly to the student using second person."
            ),
        },
        {"role": "user", "content": build_prompt(record)},
    ]
    return generate_text(
        model,
        tokenizer,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate feedback with the trained SFT/DPO adapter model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--feedback-field", default=DEFAULT_FEEDBACK_FIELD)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output and regenerate.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    if args.overwrite and args.output.exists():
        args.output.unlink()

    records = read_jsonl(args.input)
    existing_ids = load_existing_ids(args.output)
    model, tokenizer = load_model(args.base_model, args.adapter)

    generated_count = 0
    for index, record in enumerate(records):
        if args.limit is not None and generated_count >= args.limit:
            break
        if record.get("id") in existing_ids:
            continue
        if not record.get("student_answer"):
            raise ValueError(f"Missing student_answer for {record.get('id')}.")

        print(f"[{index + 1}/{len(records)}] Generating new model feedback for {record.get('id')}")
        feedback = generate_feedback(
            model,
            tokenizer,
            record,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        output_record = {
            **record,
            args.feedback_field: feedback,
            "new_model_base_model": args.base_model,
            "new_model_adapter": str(args.adapter),
        }
        append_jsonl(output_record, args.output)
        existing_ids.add(record["id"])
        generated_count += 1

    print(f"Generated {generated_count} new model feedback records.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
