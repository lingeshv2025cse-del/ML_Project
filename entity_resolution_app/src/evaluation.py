from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


LABEL_CANDIDATES = {"match", "is_match", "label", "target", "match_label", "entity_match"}


def detect_label_columns(frame: pd.DataFrame):
    detected = []
    for column in frame.columns:
        normalized = str(column).strip().lower()
        if normalized in LABEL_CANDIDATES:
            detected.append(column)
    return detected


def evaluate_predictions(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    else:
        metrics["roc_auc"] = None
    return metrics
