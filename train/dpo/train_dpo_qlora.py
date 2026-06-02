import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "train" / "dpo" / "dpo_dataset.jsonl"
DEFAULT_SFT_ADAPTER = ROOT_DIR / "train" / "model" / "sft_adapter"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "train" / "model" / "dpo_adapter"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MAX_PROMPT_LENGTH = 1536  # Maximum token length for the prompt portion of each DPO example.
DEFAULT_MAX_LENGTH = 2048  # Maximum total token length for prompt plus chosen/rejected response.
DEFAULT_EPOCHS = 1.0  # Number of full passes over the DPO dataset.
DEFAULT_BATCH_SIZE = 1  # Per-device batch size used during DPO training.
DEFAULT_GRAD_ACCUM = 8  # Number of mini-batches accumulated before one optimizer step.
DEFAULT_LEARNING_RATE = 5e-5  # Step size for DPO adapter updates.
DEFAULT_BETA = 0.1  # DPO preference strength; higher values enforce preferences more strongly.
DEFAULT_SAVE_STEPS = 200  # Save a checkpoint every N optimizer steps.
DEFAULT_LOGGING_STEPS = 10  # Log training metrics every N optimizer steps.

LORA_R = 16  # Rank of the LoRA update matrices when no SFT adapter is provided.
LORA_ALPHA = 32  # Scaling factor applied to LoRA updates.
LORA_DROPOUT = 0.05  # Dropout applied inside LoRA layers to reduce overfitting.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Transformer modules adapted by LoRA.


def load_tokenizer(model_name, adapter_path):
    tokenizer_source = adapter_path if adapter_path.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run DPO with QLoRA for the interview coach model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sft-adapter", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model, args.sft_adapter)
    dataset = load_dataset("json", data_files=str(args.data), split="train")

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.sft_adapter.exists():
        model = PeftModel.from_pretrained(model, str(args.sft_adapter), is_trainable=True)
    model.config.use_cache = False

    peft_config = None
    if not args.sft_adapter.exists():
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        )

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved DPO adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
