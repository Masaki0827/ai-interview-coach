import csv
import json
import os

def convert_csv_to_json(csv_path, json_output_path):
    data = []
    # Using utf-8-sig to handle BOM and errors='replace' for invalid chars
    with open(csv_path, mode='r', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "id": row.get("Question Number", ""),
                "question": row.get("Question", ""),
                "reference_answer": row.get("Answer", ""),
                "category": row.get("Category", ""),
                "difficulty": row.get("Difficulty", "")
            })
    
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Converted {len(data)} questions to {json_output_path}")

if __name__ == "__main__":
    csv_file = "/Users/jackie/Desktop/osu/2026 spring/ST NLP WITH DEEP LEARNING AI_539/final project/final/Software Questions.csv"
    output_file = "/Users/jackie/Desktop/osu/2026 spring/ST NLP WITH DEEP LEARNING AI_539/final project/final/data/questions.json"
    convert_csv_to_json(csv_file, output_file)
