"""Reproducible educational heart disease classification; not medical advice."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

SOURCE_URL = "https://raw.githubusercontent.com/PacktWorkshops/The-Data-Analysis-Workshop/master/Chapter07/Dataset/heart.csv"

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
            "exang","oldpeak","slope","ca","thal"]

def load_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = set([*FEATURES,"target"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[[*FEATURES,"target"]].drop_duplicates().copy()
    if df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Dataset has missing/non-finite values")
    if set(df["target"].unique()) != {0,1}:
        raise ValueError("Expected binary 0/1 target")
    return df

def metrics(y, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y,predictions)),
        "precision": float(precision_score(y,predictions,zero_division=0)),
        "recall": float(recall_score(y,predictions,zero_division=0)),
        "f1": float(f1_score(y,predictions,zero_division=0)),
        "roc_auc": float(roc_auc_score(y,probabilities)),
        "confusion_matrix": confusion_matrix(y,predictions,labels=[0,1]).tolist(),
        "classification_report": classification_report(y,predictions,output_dict=True,zero_division=0)
    }

def evaluate(df, output_dir):
    X,y = df[FEATURES],df["target"].astype(int)
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.20,stratify=y,random_state=42)
    pipeline = Pipeline([("scale",StandardScaler()),
                         ("logistic",LogisticRegression(max_iter=2000,random_state=42))])
    pipeline.fit(X_train,y_train)
    test_prob = pipeline.predict_proba(X_test)[:,1]
    test_pred = pipeline.predict(X_test)
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    cv_prob = cross_val_predict(pipeline,X,y,cv=cv,method="predict_proba")[:,1]
    cv_pred = (cv_prob>=0.5).astype(int)
    report = {
        "data_rows":len(df),"positive_cases":int(y.sum()),"training_rows":len(y_train),
        "test_rows":len(y_test),
        "holdout":metrics(y_test,test_pred,test_prob),
        "five_fold_out_of_fold":metrics(y,cv_pred,cv_prob),
        "protocol":"Duplicate removal; stratified 80/20 holdout; training-only pipeline scaler; shuffled stratified 5-fold out-of-fold evaluation"
    }
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    pd.DataFrame({"actual":y_test.to_numpy(),"predicted":test_pred,
                  "probability":test_prob}).to_csv(out/"holdout_predictions.csv",index=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ("holdout", "five_fold_out_of_fold")}, indent=2))
    print("Holdout metrics:", json.dumps({k: v for k, v in report["holdout"].items() if k != "classification_report"}, indent=2))
    print("5-fold CV metrics:", json.dumps({k: v for k, v in report["five_fold_out_of_fold"].items() if k != "classification_report"}, indent=2))
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    parser.add_argument("--data",default=SOURCE_URL, help="Local CSV or README source URL")
    parser.add_argument("--out",default=root/"results")
    args=parser.parse_args()
    evaluate(load_dataset(args.data),args.out)
