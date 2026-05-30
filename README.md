# AI Interview Coach - Qwen2.5 project

AI-driven interview simulation and feedback system using Qwen2.5.

## Roadmap
- [x] Dataset Building & Preparation
- [x] Student Answer Generation (Test Set)
- [x] Baseline Feedback Generation (Test Set)
- [ ] Baseline Feedback Scoring with Qwen3.6 (Judge)
- [ ] Preference Candidate Generation (RLAIF - Training Set)
- [ ] Preference Pair Selection with Qwen3.6 (RLAIF - Training Set)
- [ ] SFT & DPO Training with QLoRA (RLAIF)
- [ ] New Model Feedback Generation & Evaluation
- [ ] Baseline vs New Model Comparison

## Model Roles

- **Base/Policy Model**: `Qwen/Qwen2.5-3B-Instruct`
  - Used for student answer simulation and generating coach feedback candidates.
  - This is the model that will be fine-tuned.
- **Judge Model**: `Qwen/Qwen3.6-35B-A3B`
  - Used for feedback scoring and preference selection (RLAIF).

## Methodology Details

### Student Simulation Profile
To test the coach's effectiveness, we simulate answers from a specific persona:
- **Level**: First-year CS Master's student.
- **Characteristics**: Strong fundamentals but limited industry experience; prone to omitting edge cases or using imprecise terminology.
- **Answer Types**: Correct but incomplete, partially correct, incorrect, too vague, or verbose but unfocused.

### Scoring Rubric (1-20 Scale)
The judge model evaluates feedback across five dimensions:
1. **Technical Correctness**: Accuracy of the technical advice.
2. **Specificity**: Depth of identification of strengths and gaps.
3. **Helpfulness**: Potential for student improvement.
4. **Actionability**: Clarity of next steps.
5. **Interview Coaching Quality**: Suitability for an interview context.

*Note: Scores of 18-20 are reserved for "Exceptional" feedback with concrete code snippets or master-class guidance.*

### RLAIF Feedback Styles
During training data generation, we prompt the base model with diverse styles to explore the preference space:
- Technical precision focus.
- Interview communication focus.
- Balanced coaching (strengths/weaknesses/advice).
- Supportive yet strict tone.

## Dataset Schema

### Train/Test Base Data (`data/train.jsonl`, `data/test.jsonl`)
```json
{
  "id": "train_0001",
  "question": "...",
  "reference_answer": "...",
  "category": "...",
  "difficulty": "...",
  "student_answer": "...",
  "student_answer_type": "...",
  "source": "..."
}
```

### Baseline Feedback (`baseline/baseline_outputs.jsonl`)
Generated from the test set using the base model.
```json
{
  "id": "test_0001",
  "baseline_feedback": "...",
  "baseline_model": "Qwen/Qwen2.5-3B-Instruct",
  "...": "base_fields"
}
```

### Feedback Scoring (`baseline/baseline_scores.jsonl`)
Scored by the judge model on a 1-20 scale.
```json
{
  "id": "test_0001",
  "technical_correctness": 15,
  "specificity": 14,
  "helpfulness": 16,
  "actionability": 15,
  "interview_coaching_quality": 15,
  "overall_score": 15.0,
  "reason": "Concise and technically accurate with clear improvement steps.",
  "judge_model": "Qwen/Qwen3.6-35B-A3B",
  "feedback_field": "baseline_feedback"
}
```

## Workflow & Commands

### 1. Setup
Install dependencies and verify model access:
```bash
pip install -r requirements.txt
python setup_model.py
```

### 2. Baseline Evaluation
Score the existing baseline feedback to establish a benchmark:
```bash
python eval/score_feedback.py \
    --input baseline/baseline_outputs.jsonl \
    --output baseline/baseline_scores.jsonl \
    --feedback-field baseline_feedback
```

### 3. RLAIF: Preference Data Generation
Generate candidate pairs for the training set:
```bash
python train/generate_candidates.py \
    --input data/train.jsonl \
    --output train/preference_candidates.jsonl
```

### 4. RLAIF: Preference Labeling
Use the judge model to select the better candidate:
```bash
python eval/score_preferences.py \
    --input train/preference_candidates.jsonl \
    --output train/preference_pairs.jsonl
```

### 5. Fine-Tuning (Upcoming)
Fine-tune the model using the generated preference pairs (DPO).
