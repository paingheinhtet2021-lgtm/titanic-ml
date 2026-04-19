# Titanic Survival Prediction

A study in why simple wins on small data. Four submissions, four lessons.

Best public LB: **0.7656** — achieved by a single-line rule, beating all ML approaches including a heavily tuned XGBoost ensemble.

## Results

| Submission | 5-fold CV | Public LB |
|---|---|---|
| XGBoost + feature engineering + Optuna + target encoding | 0.8440 | 0.7536 |
| **Naive rule: "all females survive"** | — | **0.7656** |
| Simple XGBoost, 10 raw features, no tuning | 0.8373 | 0.7584 |
| Seed averaging + rule overrides + family-survival lookup | — | 0.7273 |

## The Full Story

### Attempt 1 — The "serious" ML pipeline

Built a textbook tabular pipeline: Title/FamilySize/CabinDeck/interaction features, out-of-fold target encoding, 60-trial Optuna search over XGBoost, stratified 5-fold CV. Achieved a strong CV of 0.8440. Submitted → **0.7536**. A 9-point CV-LB gap meant something was wrong.

### Attempt 2 — The sanity check

Submitted the most trivial possible rule: every female survives, every male dies. No ML. **0.7656** — beats the serious pipeline by 1.2 points.

### Attempt 3 — Strip the complexity

Rebuilt with 10 raw features, `max_depth=3`, default hyperparameters, no Optuna. Hypothesis: heavy tuning on small CV folds overfits fold-specific noise. CV 0.8373, LB 0.7584 — much tighter gap, hypothesis confirmed.

### Attempt 4 — Domain knowledge override

Seed averaging (20 seeds) + "women and children first" rule overrides + family-survival lookup using training-set group outcomes. **0.7273** — the family override introduced systematic errors on small families where single-member training labels didn't predict relatives reliably.

## Lessons

**1. Complex ML loses to trivial baselines on small data.** 891 training rows is not enough to resolve subtle patterns reliably. Sex alone captures ~77% of predictable variance.

**2. Hyperparameter search overfits CV folds.** 60 Optuna trials on 170-row validation sets find configurations that exploit fold-specific noise. CV 0.844 → LB 0.754 is the evidence.

**3. Approximate rules can hurt more than they help.** The family-survival override was correct on average but wrong per-row often enough to degrade the score.

**4. Diagnose CV-LB gaps; they're the main overfitting signal.** Normal gap is ~0.03. A 9-point gap means CV is wrong, not LB — always ablate and rebuild.

## What I'd do differently

- Start with the simplest possible baseline **first**, not last.
- Cap Optuna at 10–20 trials on small datasets, or skip it entirely.
- Spend small-data time on understanding the data, not tuning the model.

## Repository Structure
├── notebook.ipynb        # Full pipeline with all four attempts and diagnostics
├── submission.csv        # Best submission (0.7656 naive rule)
├── figures/              # EDA, confusion matrix, feature importance, calibration
└── requirements.txt

## Setup

```bash
pip install -r requirements.txt
kaggle competitions download -c titanic
unzip titanic.zip
jupyter notebook notebook.ipynb
```

## Tech stack

pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, matplotlib, seaborn.
