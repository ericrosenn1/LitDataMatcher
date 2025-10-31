#!/usr/bin/env python3
"""
train_future_classifier.py
Trains the Option A "strong future-direction" classifier used by lit_analyzer.py
"""

import os, argparse, json, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

# --- import helpers from your analyzer (same dir) ---------------------------
from lit_analyzer import ensure_sentence_embedder, build_future_features

# ---------------------------------------------------------------------------
def read_labeled_examples(path: str):
    """Reads either CSV or JSONL into [(sentence, label)]"""
    examples = []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s = row.get("sentence") or row.get("text") or ""
                y = row.get("label") or row.get("requires_new_data") or "0"
                try:
                    label = int(float(y))
                except Exception:
                    label = 0
                if s.strip():
                    examples.append((s.strip(), label))
    else:  # JSONL
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    s = j.get("sentence") or j.get("text") or ""
                    label = int(j.get("label") or j.get("requires_new_data") or 0)
                    if s.strip():
                        examples.append((s.strip(), label))
                except Exception:
                    pass
    return examples


def train_classifier(examples, out_path="models/future_strong.joblib"):
    print(f"[INFO] Building embeddings for {len(examples)} examples...")

    # sentences + labels
    sents  = [s for s, _ in examples]
    labels = np.array([y for _, y in examples], dtype=int)

    # IMPORTANT: build all features in one batch call
    X = build_future_features(sents)   # shape: (n_samples, n_features)
    y = labels

    # split / fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf  = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds, digits=3))
    print(f"AUC: {roc_auc_score(y_test, probs):.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dump(clf, out_path)
    print(f"[OK] Model saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to labeled CSV or JSONL with fields sentence,text,label")
    ap.add_argument("--out", default="models/future_strong.joblib", help="Output model path")
    args = ap.parse_args()

    examples = read_labeled_examples(args.data)
    if not examples:
        print("No examples found — check your file format.")
        return

    train_classifier(examples, args.out)


if __name__ == "__main__":
    main()