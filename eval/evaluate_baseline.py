import json
import torch
import gc
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import re

# =====================================================================
# CONFIGURATION
# =====================================================================
# The "Student": Qwen2.5-0.5B-Instruct (The model we are testing)
STUDENT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# The "Professor": Qwen3.6-35B-A3B (The "Higher Level" Judge)
PROFESSOR_MODEL = "Qwen/Qwen3.6-35B-A3B"

class EvaluationSystem:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.processor = None

    def load_model(self, model_name, is_multimodal=False):
        """Loads a model and its processor/tokenizer, clearing memory first."""
        self.unload_model()
        print(f"\n--- Loading Model: {model_name} ---")
        
        if is_multimodal:
            # Qwen 3.6-35B-A3B requires AutoProcessor and AutoModelForImageTextToText
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        else:
            # Qwen 2.5 uses standard CausalLM and Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )

    def unload_model(self):
        """Frees memory to allow loading larger models."""
        if self.model is not None:
            del self.model
            if self.tokenizer: del self.tokenizer
            if self.processor: del self.processor
            self.model = None
            self.tokenizer = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def generate(self, prompt, system_message="You are a helpful assistant.", max_tokens=512, is_multimodal=False):
        """Generates text from the currently loaded model."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        if is_multimodal:
            # Multi-modal prompt formatting for Qwen 3.6-35B
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device)
            
            ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            output_text = self.processor.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            # Remove <think> tags if they exist to extract the final judgement
            output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
            return output_text
        else:
            # Standard text prompt formatting for Qwen 2.5
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            ids = self.model.generate(**inputs, max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    # PATHS
    data_path = "/Users/jackie/Desktop/osu/2026 spring/ST NLP WITH DEEP LEARNING AI_539/final project/final/data/questions.json"
    final_path = "/Users/jackie/Desktop/osu/2026 spring/ST NLP WITH DEEP LEARNING AI_539/final project/final/eval/baseline_results.json"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, 'r') as f:
        questions = json.load(f)
    
    sample = questions[:10]
    eval_sys = EvaluationSystem()

    # Pass 1: Student Generation (Local Qwen 2.5)
    eval_sys.load_model(STUDENT_MODEL, is_multimodal=False)
    for item in tqdm(sample, desc="Student Testing"):
        ans = eval_sys.generate(f"Question: {item['question']}\nConcise answer:", system_message="You are a candidate.")
        item['baseline_answer'] = ans

    # Pass 2: Professor Judgement (Higher-level Qwen 3.6)
    eval_sys.load_model(PROFESSOR_MODEL, is_multimodal=True)
    total_score = 0
    for item in tqdm(sample, desc="Professor Grading"):
        judgement_prompt = f"Q: {item['question']}\nRef: {item['reference_answer']}\nAns: {item['baseline_answer']}\nScore (0-10):"
        score_str = eval_sys.generate(judgement_prompt, system_message="You are a senior judge.", max_tokens=200, is_multimodal=True)
        
        # Parse the score
        match = re.search(r"([0-9]*\.?[0-9]+)", score_str)
        score = float(match.group(1)) if match else 5.0
        item['accuracy_score'] = min(max(score, 0), 10)
        total_score += item['accuracy_score']

    # Report
    avg_score = total_score / len(sample)
    report = {"student": STUDENT_MODEL, "professor": PROFESSOR_MODEL, "average_score": avg_score, "results": sample}
    
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with open(final_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nAverage Accuracy: {avg_score:.2f} / 10.0")

if __name__ == "__main__":
    main()
