import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "baseline" / "baseline_outputs.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "baseline" / "baseline_scores.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

RUBRIC_FIELDS = [
    "technical_correctness",
    "specificity",
    "helpfulness",
    "actionability",
    "interview_coaching_quality",
]


def read_jsonl(path):
    records = []
    if not path.exists(): return []
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
    return {record["id"] for record in read_jsonl(path) if "id" in record}


def load_model(model_name, quantize=False):
    print(f"Loading judge model: {model_name} (quantize={quantize})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def generate_text(model, tokenizer, messages, max_new_tokens=1024):
    """Uses assistant pre-filling to skip thinking process and force JSON output."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # FORCE PRE-FILL: Append the start of a JSON object
    prompt += "{"
    
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, # Greedy decoding
        pad_token_id=tokenizer.eos_token_id,
    )
    
    new_ids = output_ids[0][inputs.input_ids.shape[1] :]
    # Prepend the forced "{" back to the result
    return "{" + tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def extract_json_object(text):
    # 1. Aggressively strip any model-specific tags that might be in the middle of the response
    text = re.sub(r"<\/?think>", "", text, flags=re.IGNORECASE).strip()

    # 2. Search for all blocks that look like JSON objects {...}
    # Using [^{}]* inside the braces to find discrete blocks
    matches = list(re.finditer(r"\{[^{}]*\"[^{}]*:[^{}]*\}", text, flags=re.DOTALL))

    if not matches:
        # Fallback: if discrete blocks fail, try to find the last valid-looking string between braces
        start = text.rfind('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or start > end:
            raise ValueError(f"No JSON object found in response: {text[:200]}")
        json_str = text[start : end + 1]
    else:
        # Take the LAST match (the official answer after any thinking/drafting)
        json_str = matches[-1].group(0)

    json_str = json_str.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Heuristic fix for single quotes and trailing commas
        fixed = json_str.replace("'", '"')
        fixed = re.sub(r",\s*\}", "}", fixed)
        try:
            return json.loads(fixed)
        except Exception as e:
            print(f"\n[!] Final JSON parse failure. Content:\n{json_str}\n")
            raise e



def build_prompt(record, feedback_field):
    return f"""Evaluate the coach feedback for a software engineering interview answer.

Question:
{record.get("question", "")}

Reference answer:
{record.get("reference_answer", "")}

Student answer:
{record.get("student_answer", "")}

Coach feedback:
{record.get(feedback_field, "")}

Score the coach feedback from 1 to 20 for each dimension:
- technical_correctness: Is the feedback technically accurate?
- specificity: Does it point to specific strengths, mistakes, or missing details?
- helpfulness: Wood it help the student improve?
- actionability: Does it give concrete next steps?
- interview_coaching_quality: Does it sound like useful interview coaching instead of generic commentary?

Scoring calibration (1 to 20 scale):
- 10 to 12 (Standard / Good): The feedback is normally good, accurate, and generally useful, but lacks deep technical depth or highly tailored guidance.
- 15 to 17 (Clearly Strong): The feedback is highly precise, technically accurate, identifies specific gaps, and provides clear actionable improvements.
- 18 to 20 (Exceptional / Near-Perfect): Reserved ONLY for master-class, exceptional coaching. Give this ONLY when the feedback is flawless, extremely precise, and includes concrete examples, tailored code snippets, or explicit next-step practice guidance.

CRITICAL CONSTRAINTS TO AVOID LENIENCY BIAS:
1. Do NOT give 18-20 for feedback that is merely good, polite, or generally helpful.
2. If you find and mention ANY "minor inaccuracies", "gaps", "opportunities for deeper elaboration", or "flaws" in your 'reason', you MUST NOT give a score higher than 14 for that specific dimension. 
3. Be highly critical. A score of 20 means there is absolutely ZERO room for improvement.

Return only valid JSON in this exact format:
{{
  "technical_correctness": 1,
  "specificity": 1,
  "helpfulness": 1,
  "actionability": 1,
  "interview_coaching_quality": 1,
  "overall_score": 1.0,
  "reason": "..."
}}
"""


def normalize_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 10.0
    return min(max(score, 1.0), 20.0)


def score_feedback(model, tokenizer, record, feedback_field):
    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON-only evaluator. No thoughts, no reasoning, just output the JSON object.",
        },
        {"role": "user", "content": build_prompt(record, feedback_field)},
    ]
    raw_output = generate_text(model, tokenizer, messages)
    
    try:
        parsed = extract_json_object(raw_output)
    except Exception as e:
        print(f"\n--- DEBUG: RAW LLM RESPONSE (PARSING FAILED) ---\n{raw_output}\n--- END DEBUG ---\n")
        raise e

    scores = {field: normalize_score(parsed.get(field)) for field in RUBRIC_FIELDS}
    overall = parsed.get("overall_score")
    if overall is None:
        overall = sum(scores.values()) / len(scores)
    else:
        overall = normalize_score(overall)

    return {
        "id": record.get("id"),
        **scores,
        "overall_score": round(float(overall), 2),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Score coach feedback with a judge model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--feedback-field", default="baseline_feedback")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization.")
    args = parser.parse_args()

    if args.overwrite and args.output.exists():
        args.output.unlink()

    records = read_jsonl(args.input)
    existing_ids = load_existing_ids(args.output)
    model, tokenizer = load_model(args.model, quantize=args.quantize)

    scored_count = 0
    for index, record in enumerate(records):
        if args.limit is not None and scored_count >= args.limit:
            break
        if record.get("id") in existing_ids:
            continue
        if not record.get(args.feedback_field):
            raise ValueError(f"Missing {args.feedback_field} for {record.get('id')}")

        print(f"[{index + 1}/{len(records)}] Scoring {record.get('id')}...")
        try:
            score_record = score_feedback(model, tokenizer, record, args.feedback_field)
            score_record["judge_model"] = args.model
            score_record["feedback_field"] = args.feedback_field
            append_jsonl(score_record, args.output)
            existing_ids.add(record["id"])
            scored_count += 1
        except Exception as e:
            print(f"  [!] Failed to score {record.get('id')}: {e}")

    print(f"Scored {scored_count} records.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
