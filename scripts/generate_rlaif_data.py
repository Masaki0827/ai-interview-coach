import json
import torch
import gc
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
import os
import re

# =====================================================================
# CONFIGURATION
# =====================================================================
# We use the 35B model we just downloaded to generate high-quality preference data.
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

class RLAIFGenerator:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading model for RLAIF data generation: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Use bfloat16 to save memory during loading if supported, otherwise float16
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        kwargs = {
            "torch_dtype": compute_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Force everything onto the first GPU if available
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
            
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **kwargs
        )

    def generate(self, prompt, system_message="You are a helpful assistant.", max_tokens=512):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        output_text = self.processor.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Remove thinking process tags
        output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
        return output_text

def main():
    data_path = "/Users/jackie/Desktop/osu/2026 spring/ST NLP WITH DEEP LEARNING AI_539/final project/final/data/questions.json"
    output_path = "/Users/jackie/Desktop/osu/2026 spring/ST NLP WITH DEEP LEARNING AI_539/final project/final/data/rlaif_train_data.json"

    with open(data_path, 'r') as f:
        questions = json.load(f)
    
    # We'll generate 20 pairs for this run
    sample = questions[:20]
    generator = RLAIFGenerator()
    
    rlaif_data = []
    print("\n--- Generating RLAIF Preference Pairs ---")
    
    for item in tqdm(sample):
        question = item['question']
        ref = item['reference_answer']
        
        # 1. Generate a "Student Answer"
        student_ans = generator.generate(
            f"Question: {question}\nReference Answer: {ref}\nAct as a student and provide a conversational answer.",
            system_message="You are a software engineering student."
        )
        
        # 2. Generate "Chosen" Feedback (Helpful, Specific, Encouraging)
        chosen_feedback = generator.generate(
            f"Question: {question}\nStudent Answer: {student_ans}\nReference: {ref}\nProvide high-quality, encouraging, and technically detailed feedback as a mentor.",
            system_message="You are an expert AI Interview Coach."
        )
        
        # 3. Generate "Rejected" Feedback (Vague, discouraging, or too brief)
        rejected_feedback = generator.generate(
            f"Question: {question}\nStudent Answer: {student_ans}\nProvide vague and slightly discouraging feedback on this answer.",
            system_message="You are a lazy and overly critical interviewer."
        )
        
        rlaif_data.append({
            "prompt": f"Question: {question}\nStudent Answer: {student_ans}",
            "chosen": chosen_feedback,
            "rejected": rejected_feedback
        })

    with open(output_path, 'w') as f:
        json.dump(rlaif_data, f, indent=2)
    
    print(f"\nSuccess! Generated {len(rlaif_data)} RLAIF training pairs at: {output_path}")

if __name__ == "__main__":
    main()
