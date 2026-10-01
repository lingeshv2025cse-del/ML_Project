from __future__ import annotations

from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def _build_classifier():
    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="binary:logistic",
            use_label_encoder=False,
            eval_metric="logloss",
        )
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
    )


def train_model(feature_frame: pd.DataFrame, target_values: pd.Series):
    model = _build_classifier()
    model.fit(feature_frame, target_values)
    return model


def save_model(model, model_path: str) -> None:
    joblib.dump(model, model_path)


def load_model(model_path: str):
    return joblib.load(model_path)


def generate_weak_labels(candidate_features: List[Dict[str, float]]) -> Tuple[pd.DataFrame, pd.Series, str]:
    rows = []
    labels = []
    for item in candidate_features:
        name_similarity = float(item.get("name_similarity", 0.0))
        phone_similarity = float(item.get("phone_similarity", 0.0))
        email_similarity = float(item.get("email_similarity", 0.0))
        website_similarity = float(item.get("website_similarity", 0.0))
        exact_phone = float(item.get("exact_phone_match", 0.0))
        exact_email = float(item.get("exact_email_match", 0.0))
        exact_name = float(item.get("exact_name_match", 0.0))

        is_positive = (
            (exact_phone == 1.0 and name_similarity >= 0.70)
            or (exact_email == 1.0 and name_similarity >= 0.60)
            or (website_similarity >= 0.90 and name_similarity >= 0.75)
            or (exact_name == 1.0 and phone_similarity >= 0.80)
        )
        is_negative = (
            name_similarity <= 0.25 and phone_similarity <= 0.20 and email_similarity <= 0.20 and website_similarity <= 0.20
        )

        if is_positive:
            rows.append(item)
            labels.append(1)
        elif is_negative:
            rows.append(item)
            labels.append(0)

    if not rows:
        rows = [
            {"name_similarity": 0.95, "address_similarity": 0.90, "phone_similarity": 1.0, "email_similarity": 1.0, "website_similarity": 0.95, "city_similarity": 0.95, "postal_similarity": 1.0, "exact_name_match": 1.0, "exact_phone_match": 1.0, "exact_email_match": 1.0, "exact_postal_match": 1.0},
            {"name_similarity": 0.10, "address_similarity": 0.05, "phone_similarity": 0.05, "email_similarity": 0.0, "website_similarity": 0.0, "city_similarity": 0.05, "postal_similarity": 0.0, "exact_name_match": 0.0, "exact_phone_match": 0.0, "exact_email_match": 0.0, "exact_postal_match": 0.0},
        ]
        labels = [1, 0]

    if len(set(labels)) < 2:
        synthetic_positive = {"name_similarity": 0.95, "address_similarity": 0.90, "phone_similarity": 1.0, "email_similarity": 1.0, "website_similarity": 0.95, "city_similarity": 0.95, "postal_similarity": 1.0, "exact_name_match": 1.0, "exact_phone_match": 1.0, "exact_email_match": 1.0, "exact_postal_match": 1.0}
        synthetic_negative = {"name_similarity": 0.10, "address_similarity": 0.05, "phone_similarity": 0.05, "email_similarity": 0.0, "website_similarity": 0.0, "city_similarity": 0.05, "postal_similarity": 0.0, "exact_name_match": 0.0, "exact_phone_match": 0.0, "exact_email_match": 0.0, "exact_postal_match": 0.0}
        if 1 not in set(labels):
            rows.append(synthetic_positive)
            labels.append(1)
        if 0 not in set(labels):
            rows.append(synthetic_negative)
            labels.append(0)

    feature_frame = pd.DataFrame(rows)
    for key in ["name_similarity", "address_similarity", "phone_similarity", "email_similarity", "website_similarity", "city_similarity", "postal_similarity", "exact_name_match", "exact_phone_match", "exact_email_match", "exact_postal_match"]:
        feature_frame[key] = feature_frame.get(key, 0.0).fillna(0.0)

    return feature_frame, pd.Series(labels), "Weak labels generated automatically from strong similarity rules."


def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

    if y_prob is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics
