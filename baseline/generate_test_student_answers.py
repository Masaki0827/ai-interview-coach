import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "test.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "test.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ANSWER_TYPES = {
    "correct_but_incomplete",
    "partially_correct",
    "incorrect",
    "too_vague",
    "verbose_but_unfocused",
}


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records, path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def load_model(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


def generate_text(model, tokenizer, messages, max_new_tokens=256, temperature=0.7, top_p=0.9):
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
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def build_prompt(record):
    return f"""Create a realistic simulated student answer for a CS/software engineering interview question.

Student profile:
- The student is a first-year master's student in Computer Science.
- They have completed undergraduate-level CS coursework.
- They understand many core CS concepts but may not explain them precisely.
- They have limited industry software engineering interview experience.
- They may omit edge cases, tradeoffs, examples, or exact terminology.
- Their answer should sound natural, slightly uncertain, and imperfect.
- Their answer should not sound like a textbook, professor, or reference solution.
- Their answer should often be incomplete, slightly imprecise, and limited to a few spoken-style sentences.
- Do not mention that this is a simulated answer.

The student answer should sound like a real interview response, not a reference answer.

Answer length:
- 2 to 4 sentences only.

Realism rules:
- Do not copy the reference answer.
- Do not cover every important point.
- Usually miss at least one important detail, edge case, tradeoff, or precise term.
- Use natural interview-style wording such as "I think", "basically", "maybe", or "I'm not totally sure".
- Avoid polished textbook-style explanations.
- Do not use bullet points or numbered lists.

Choose exactly one answer type:
- correct_but_incomplete
- partially_correct
- incorrect
- too_vague
- verbose_but_unfocused

Question:
{record["question"]}

Reference answer:
{record["reference_answer"]}

Return only valid JSON in this format:
{{
  "student_answer": "...",
  "student_answer_type": "partially_correct"
}}
"""


def generate_student_answer(model, tokenizer, record):
    messages = [
        {
            "role": "system",
            "content": (
                "You simulate realistic first-year master's students in Computer Science "
                "answering software engineering interview questions. The student has solid "
                "CS fundamentals but limited industry interview experience. Their answers "
                "should be natural, sometimes incomplete, slightly imprecise, and not overly polished."
            ),
        },
        {"role": "user", "content": build_prompt(record)},
    ]
    raw_output = generate_text(model, tokenizer, messages)
    parsed = extract_json_object(raw_output)

    student_answer = str(parsed.get("student_answer", "")).strip()
    answer_type = str(parsed.get("student_answer_type", "")).strip()

    if not student_answer:
        raise ValueError("Model returned an empty student_answer.")
    if answer_type not in ANSWER_TYPES:
        answer_type = "partially_correct"

    return student_answer, answer_type


def main():
    parser = argparse.ArgumentParser(description="Generate fixed student answers for the held-out test set.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing student answers.")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    model, tokenizer = load_model(args.model)

    generated_count = 0
    for index, record in enumerate(records):
        if args.limit is not None and generated_count >= args.limit:
            break
        if record.get("student_answer") and not args.overwrite:
            continue

        print(f"[{index + 1}/{len(records)}] Generating student answer for {record.get('id')}")
        try:
            student_answer, answer_type = generate_student_answer(model, tokenizer, record)
            record["student_answer"] = student_answer
            record["student_answer_type"] = answer_type
            generated_count += 1
        except Exception as exc:
            print(f"Failed on {record.get('id')}: {exc}")
            record["student_answer"] = ""
            record["student_answer_type"] = "generation_failed"

        write_jsonl(records, args.output)

    print(f"Generated {generated_count} student answers.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
