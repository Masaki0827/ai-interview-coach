# Qwen2.5 Setup and Inference Script for AI Interview Coach
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful AI Interview Coach."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    model, tokenizer = load_model()
    prompt = "Give me a technical interview question about Python decorators."
    print(f"User: {prompt}")
    response = generate_response(model, tokenizer, prompt)
    print(f"AI Coach: {response}")
