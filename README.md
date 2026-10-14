# Heart Disease Prediction — Leakage-Aware Machine Learning

**Python | Pandas | Scikit-learn | Logistic Regression | Stratified Validation**

An educational binary classification study of heart-disease labels from tabular clinical features. **Not a medical diagnostic or clinical decision-making system.**

## Overview
The original project includes exploratory data analysis, duplicate removal, visualization, scaled Logistic Regression, a stratified holdout, a confusion matrix and an ROC curve. The source notebook is preserved unchanged. A portable and reproducible companion pipeline has now been added.

## Source Dataset
The original notebook downloads a tabular `heart.csv` dataset from the [Packt Data Analysis Workshop repository](https://github.com/PacktWorkshops/The-Data-Analysis-Workshop/blob/master/Chapter07/Dataset/heart.csv). The notebook reports **302 records after duplicate removal** (14 columns: 13 features plus binary `target`), with no missing values at inspection.

**The CSV is not copied into this public repository.** The evaluation script loads the same public source URL used by the original notebook by default (requires internet). You can also download the CSV and pass `--data data/heart.csv` to run offline. Please check the source's usage terms.

Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca and thal. This dataset is educational; label semantics and clinical utility are not independently established.

## Original Notebook: Verified Recorded Result
The original `notebooks/original_heart_disease_prediction.ipynb` reported **78.69% holdout accuracy (48/61 correct)** using an 80/20 stratified split and Logistic Regression.

**Important correction:** The old README stated **“~85%+ accuracy” and “~0.90+ AUC”**, but the notebook's recorded output explicitly shows **78.69% accuracy**, and does not provide a numeric ROC-AUC output in the inspected notebook. Those approximate headline claims have been removed.

## Improvements
- All preprocessing and Logistic Regression within a **single scikit-learn Pipeline**.
- Explicit fixed-seed, stratified 80/20 train/test split.
- StandardScaler fitted exclusively on training data within each split or fold.
- Added shuffled **5-fold stratified out-of-fold evaluation** with accuracy, precision, recall, F1, ROC-AUC and confusion matrices.
- Consistent duplicate handling, target checks and input validation.
- Export `results/evaluation.json` and `results/holdout_predictions.csv` upon local execution.
- Preserve original EDA notebook, plots and historical outputs unchanged.

## Execution and Results
The original notebook's recorded holdout accuracy is **78.69%**. The reproducible companion code has been reviewed and published, and now downloads the same original dataset URL automatically. **The exact new pipeline has not yet been verified end-to-end in a Python runtime**. We therefore do not report invented new CV/test scores.

For a quick experiment run:

```bash
git clone https://github.com/syblejan/Heart_Disease_prediction.git
cd Heart_Disease_prediction
python -m venv .venv
# Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python src/evaluate.py  # fetches original README-linked CSV automatically
# Optional offline run: python src/evaluate.py --data data/heart.csv
```

The generated JSON contains the precise holdout and cross-validation scores.

## Repository Structure
```text
Heart_Disease_prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── original_heart_disease_prediction.ipynb
│   └── reproducible_heart_disease.ipynb
└── src/
    └── evaluate.py
```

`data/` and generated `results/` are ignored by Git.

## Limitations
This is a small, retrospective educational classification dataset. A single train/test split is uncertain, and repeated cross-validation cannot establish medical reliability. Different clinics, population shifts, data leakage from related patients, feature availability, threshold trade-offs, model calibration and external validation all matter before any clinical use.

## Author
**Syble Jan** — BS Computer Science | Aspiring AI/ML Engineer

[GitHub profile](https://github.com/syblejan)
