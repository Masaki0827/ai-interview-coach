# AI Interview Coach - Qwen2.5

AI-driven interview simulation and feedback system using Qwen2.5.

## Roadmap
- [x] Dataset Building & Preparation
- [x] Student Answer Generation
- [x] Baseline Feedback Generation
- [ ] Baseline Feedback Scoring with Qwen3.6
- [ ] SFT Dataset Construction
- [ ] SFT with QLoRA
- [ ] Preference Candidate Generation (RLAIF)
- [ ] Preference Pair Selection with Qwen3.6 (RLAIF)
- [ ] DPO Training with QLoRA (RLAIF)
- [ ] New Model Feedback Generation
- [ ] New Model Feedback Scoring with Qwen3.6
- [ ] Baseline vs New Model Comparison
- [ ] Small-scale Human Evaluation

## Model Roles

- `Qwen/Qwen2.5-3B-Instruct`: base model used to generate simulated student answers and coach feedback. This is also the model to be fine-tuned.
- `Qwen/Qwen3.6-35B-A3B`: judge model used for feedback scoring and preference selection.

## Dataset Schema

Train and test examples use this schema:

```json
{
  "id": "train_0001",
  "question": "...",
  "reference_answer": "...",
  "category": "...",
  "difficulty": "...",
  "student_answer": "...",
  "student_answer_type": "partially_correct",
  "source": "software_questions"
}
```

Baseline feedback outputs use this schema:

```json
{
  "id": "test_0001",
  "question": "...",
  "reference_answer": "...",
  "category": "...",
  "difficulty": "...",
  "student_answer": "...",
  "student_answer_type": "partially_correct",
  "source": "...",
  "baseline_feedback": "...",
  "baseline_model": "Qwen/Qwen2.5-3B-Instruct"
}
```

Feedback scoring outputs use this schema:

```json
{
  "id": "test_0001",
  "technical_correctness": 4,
  "specificity": 4,
  "helpfulness": 4,
  "actionability": 4,
  "interview_coaching_quality": 4,
  "overall_score": 4.0,
  "reason": "...",
  "judge_model": "Qwen/Qwen3.6-35B-A3B",
  "feedback_field": "baseline_feedback"
}
```

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Test model: `python setup_model.py`
3. Process original software question data: `python scripts/process_data.py`
4. Generate simulated student answers for the test set: `python baseline/generate_test_student_answers.py --input data/test.jsonl --output data/test.jsonl`
5. Generate baseline coach feedback: `python baseline/generate_baseline_feedback.py --input data/test.jsonl --output baseline/baseline_outputs.jsonl`
6. Score baseline feedback with the judge model: `python eval/score_feedback.py --input baseline/baseline_outputs.jsonl --output baseline/baseline_scores.jsonl --feedback-field baseline_feedback`
