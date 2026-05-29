import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "train.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "train" / "preference_candidates.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RANDOM_SEED = 539
TEMPERATURE_OPTIONS = [0.45, 0.55, 0.65, 0.75]

FEEDBACK_STYLES = [
    "Focus on the most important technical correction and give one clear next step.",
    "Start with what is correct, then explain the most important missing concept.",
    "Emphasize interview communication: clarity, structure, and how to phrase the answer better.",
    "Emphasize technical precision: point out vague wording and replace it with specific terminology.",
    "Give balanced coaching: strengths, weaknesses, and one concrete improved answer direction.",
    "Use a supportive coaching tone while still being strict about technical accuracy.",
]


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


def generate_text(model, tokenizer, messages, max_new_tokens=512, temperature=0.6, top_p=0.9):
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


def build_prompt(record, style):
    return f"""You are writing coach feedback directly to a software engineering interview candidate.

Question:
{record.get("question", "")}

Reference answer:
{record.get("reference_answer", "")}

Student answer:
{record.get("student_answer", "")}

Write one version of interview coaching feedback.

Requirements:
- Address the candidate directly using "you" and "your answer".
- Do not say "your student", "the student", "the candidate", or speak to a professor/teacher.
- Identify what the candidate got right.
- Identify missing, vague, or incorrect technical details.
- Give concrete, actionable improvement advice.
- Keep a professional coaching tone.
- Do not simply rewrite the reference answer.
- Keep the feedback concise.

Feedback style for this version:
{style}
"""


def generate_feedback(model, tokenizer, record, style, temperature):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise and helpful AI interview coach for CS students. "
                "Always speak directly to the student using second person."
            ),
        },
        {"role": "user", "content": build_prompt(record, style)},
    ]
    return generate_text(
        model,
        tokenizer,
        messages,
        max_new_tokens=512,
        temperature=temperature,
        top_p=0.9,
    )


def select_candidate_settings(record):
    rng = random.Random(f"{RANDOM_SEED}:{record.get('id', '')}")
    style_a, style_b = rng.sample(FEEDBACK_STYLES, 2)
    temperature_a = rng.choice(TEMPERATURE_OPTIONS)
    temperature_b = rng.choice(TEMPERATURE_OPTIONS)

    if rng.random() < 0.5:
        style_a, style_b = style_b, style_a
        temperature_a, temperature_b = temperature_b, temperature_a

    return style_a, style_b, temperature_a, temperature_b


def build_candidate_record(
    record,
    feedback_a,
    feedback_b,
    feedback_a_style,
    feedback_b_style,
    feedback_a_temperature,
    feedback_b_temperature,
    model_name,
):
    return {
        "id": record.get("id"),
        "question": record.get("question", ""),
        "reference_answer": record.get("reference_answer", ""),
        "category": record.get("category", ""),
        "difficulty": record.get("difficulty", ""),
        "student_answer": record.get("student_answer", ""),
        "student_answer_type": record.get("student_answer_type", ""),
        "source": record.get("source", ""),
        "feedback_a": feedback_a,
        "feedback_b": feedback_b,
        "feedback_a_style": feedback_a_style,
        "feedback_b_style": feedback_b_style,
        "feedback_a_temperature": feedback_a_temperature,
        "feedback_b_temperature": feedback_b_temperature,
        "candidate_model": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate two coach feedback candidates for each training example.")
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
            raise ValueError(f"Missing student_answer for {record.get('id')}")

        print(f"[{index + 1}/{len(records)}] Generating candidates for {record.get('id')}")
        style_a, style_b, temperature_a, temperature_b = select_candidate_settings(record)
        feedback_a = generate_feedback(
            model,
            tokenizer,
            record,
            style=style_a,
            temperature=temperature_a,
        )
        feedback_b = generate_feedback(
            model,
            tokenizer,
            record,
            style=style_b,
            temperature=temperature_b,
        )

        if feedback_a.strip() == feedback_b.strip():
            fallback_styles = [style for style in FEEDBACK_STYLES if style not in {style_a, style_b}]
            style_b = fallback_styles[0] if fallback_styles else "Give an alternative coaching response with different wording and emphasis."
            temperature_b = 0.85
            feedback_b = generate_feedback(
                model,
                tokenizer,
                record,
                style=style_b,
                temperature=temperature_b,
            )

        candidate_record = build_candidate_record(
            record,
            feedback_a,
            feedback_b,
            style_a,
            style_b,
            temperature_a,
            temperature_b,
            args.model,
        )
        append_jsonl(candidate_record, args.output)
        existing_ids.add(record["id"])
        generated_count += 1

    print(f"Generated {generated_count} preference candidate records.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
