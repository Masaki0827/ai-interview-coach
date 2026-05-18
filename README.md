# AI Interview Coach - Qwen2.5

AI-driven interview simulation and feedback system using Qwen2.5.

## Roadmap
- [x] Dataset Building & Preparation
  - [x] Load initial question bank from Software Questions.csv (200 questions)
  - [ ] Generate student-like answers and mentor feedback for alignment data
- [ ] Supervised Fine-tuning (SFT)
- [ ] RLAIF / Alignment Training
- [ ] Evaluation

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Test model: `python setup_model.py`
3. Process data: `python scripts/process_data.py`
