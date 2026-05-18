import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # Initial system message to set the persona
    messages = [
        {"role": "system", "content": "You are a helpful AI Interview Coach. You help students prepare for software engineering interviews by asking questions and providing feedback."}
    ]

    print("\n--- AI Interview Coach is ready! ---")
    print("(Type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
            
        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input.strip():
            continue

        # Append user message
        messages.append({"role": "user", "content": user_input})

        # Keep context window small for the 0.5B model
        if len(messages) > 10:
            messages = [messages[0]] + messages[-9:]

        # Prepare prompt using chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the new parts
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\nAI Coach: {response}\n")
        
        # Append assistant response to history
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chat()
