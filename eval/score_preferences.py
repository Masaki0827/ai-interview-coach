import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = ROOT_DIR / "train" / "preference_candidates.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "train" / "preference_pairs.jsonl"
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


def load_model(model_name, quantize=False):
    print(f"Loading judge model: {model_name} (quantize={quantize})")
    print("TIP: If you get a TypeError or OOM, restart the Colab runtime and run:")
    print("!pip install -U transformers accelerate bitsandbytes")
    
    # Critical: set allocation config to avoid fragmentation
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Clear memory from previous runs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Use bfloat16 for computation and as the base dtype to save memory
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    kwargs = {
        "dtype": compute_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    # Use "cuda" string instead of "auto" to avoid complex accelerate dispatch hooks 
    # that cause TypeError in certain versions.
    if torch.cuda.is_available():
        kwargs["device_map"] = "cuda"
        # Optional: set max_memory if you still hit OOM during materialization
        # kwargs["max_memory"] = {0: "37GiB", "cpu": "32GiB"}
    else:
        kwargs["device_map"] = "auto"

    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def generate_text(model, tokenizer, messages, max_new_tokens=2048, temperature=0.1, top_p=0.9):
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
Do not prefer A or B because of position, length, formatting, or tone alone.
A shorter answer can be better if it is more precise and useful.
A longer answer can be worse if it is generic, repetitive, or unfocused.
Judge the substance of the feedback, not whether it matches a particular style label.

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
        "feedback_a_style": record.get("feedback_a_style", ""),
        "feedback_b_style": record.get("feedback_b_style", ""),
        "feedback_a_temperature": record.get("feedback_a_temperature", ""),
        "feedback_b_temperature": record.get("feedback_b_temperature", ""),
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
