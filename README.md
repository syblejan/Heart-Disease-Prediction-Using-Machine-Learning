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

## Verified Execution and Results

The **exact published `src/evaluate.py`** was run successfully in GitHub Actions using Python 3.11, scikit-learn and the **same original Packt CSV URL** as the initial notebook. The workflow installs dependencies, evaluates the model, and uploads `evaluation.json` plus `holdout_predictions.csv` as downloadable run artifacts.

**Dataset:** 303 raw records, **302 after removing duplicates**, 164 target-positive. Fixed-seed stratified split: **241 train / 61 holdout**. All scaling is fit within the training partition of the scikit-learn Pipeline.

| Metric | Stratified holdout (61) | 5-fold out-of-fold (302) |
|---|---:|---:|
| Accuracy | **78.69%** | **83.77%** |
| Precision (class 1) | 76.32% | 81.42% |
| Recall (class 1) | 87.88% | 90.85% |
| F1 (class 1) | 81.69% | 85.88% |
| ROC-AUC | 0.8647 | 0.8958 |

### Confusion matrices

Order = [actual class 0, actual class 1] across rows and [predicted class 0, predicted class 1] across columns.

| Holdout | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 19 | 9 |
| Actual 1 | 4 | 29 |

| 5-fold out-of-fold | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 104 | 34 |
| Actual 1 | 15 | 149 |

The holdout accuracy **78.69%** matches the preserved notebook output; the new out-of-fold cross-validation is a distinct evaluation, **not** a 5-point improvement on the identical holdout.

**Automated verification:** [GitHub Actions workflow](https://github.com/syblejan/Heart_Disease_prediction/actions/workflows/evaluate.yml) — [successful end-to-end run](https://github.com/syblejan/Heart_Disease_prediction/actions/runs/38611740132). Execution artifacts are attached to the workflow run, not committed to the repository.

Run the experiment yourself:

```bash
git clone https://github.com/syblejan/Heart_Disease_prediction.git
cd Heart_Disease_prediction
python -m venv .venv
# Activate virtual environment
python -m pip install -r requirements.txt
python src/evaluate.py
# Optional offline run: python src/evaluate.py --data data/heart.csv
```

**Caution:** This is an educational model trained on a small retrospective dataset; the scores do not establish clinical validity or safety.

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
