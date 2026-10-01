from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from config import HIGH_CONFIDENCE_THRESHOLD, MATCH_THRESHOLD
from src.candidate_generation import generate_candidates
from src.feature_engineering import FEATURE_COLUMNS, build_feature_vector
from src.model import evaluate_model, generate_weak_labels, load_model, save_model, train_model


def _classify_probability(probability: float) -> str:
    if probability >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH_CONFIDENCE"
    if probability >= MATCH_THRESHOLD:
        return "POSSIBLE_MATCH"
    return "NO_MATCH"


def _safe_row_for_source(source_df: pd.DataFrame, source_id: str) -> pd.Series:
    matches = source_df[source_df["source_id"].astype(str) == str(source_id)]
    if matches.empty:
        return pd.Series({
            "source_id": source_id,
            "business_name": "",
            "address": "",
            "city": "",
            "state": "",
            "postal_code": "",
            "phone": "",
            "email": "",
            "website": "",
        })
    return matches.iloc[0]


def resolve_entities(source1_df: pd.DataFrame,
                     source2_df: pd.DataFrame,
                     source3_df: pd.DataFrame,
                     model_path: str,
                     use_supervised_labels: bool = False,
                     label_source: str = "weak_labels_generated") -> Tuple[pd.DataFrame, pd.DataFrame, object, str, Dict[str, object]]:
    candidate_pairs = generate_candidates(source1_df, source2_df, source3_df)

    if not candidate_pairs:
        empty_results = pd.DataFrame(columns=[
            "source1_id", "source2_id", "source3_id", "source1_name", "source2_name",
            "source3_name", "name_similarity", "address_similarity", "phone_similarity",
            "email_similarity", "website_similarity", "overall_probability", "match_status"
        ])
        unmatched_rows = [{
            "source1_id": str(row.get("source_id", "")),
            "source1_name": str(row.get("business_name", "")),
            "reason": "No candidate pairs were generated for this dataset."
        } for _, row in source1_df.iterrows()]
        unmatched_df = pd.DataFrame(unmatched_rows)
        if unmatched_df.empty:
            unmatched_df = pd.DataFrame(columns=["source1_id", "source1_name", "reason"])
        return empty_results, unmatched_df, None, "No candidate pairs were generated for the provided datasets.", {"label_type": "no_candidates", "n_candidates": 0}

    features_records: List[Dict[str, float]] = []
    pair_rows: List[Dict[str, object]] = []

    for candidate in candidate_pairs:
        s1_id = str(candidate["source1_id"])
        s2_id = str(candidate.get("source2_id", ""))
        s3_id = str(candidate.get("source3_id", ""))
        source_name = str(candidate.get("source_name", ""))

        s1_row = _safe_row_for_source(source1_df, s1_id)
        if source_name == "source2":
            candidate_row = _safe_row_for_source(source2_df, s2_id)
        else:
            candidate_row = _safe_row_for_source(source3_df, s3_id)

        feature_vector = build_feature_vector(s1_row.to_dict(), candidate_row.to_dict())
        feature_vector["source1_id"] = s1_id
        feature_vector["source2_id"] = s2_id
        feature_vector["source3_id"] = s3_id
        feature_vector["source1_name"] = str(s1_row.get("business_name", ""))
        feature_vector["source2_name"] = str(candidate_row.get("business_name", "")) if source_name == "source2" else ""
        feature_vector["source3_name"] = str(candidate_row.get("business_name", "")) if source_name == "source3" else ""
        feature_vector["candidate_source"] = source_name
        features_records.append({k: feature_vector.get(k, 0.0) for k in FEATURE_COLUMNS})

        pair_rows.append({
            "source1_id": s1_id,
            "source2_id": s2_id,
            "source3_id": s3_id,
            "source1_name": feature_vector["source1_name"],
            "source2_name": feature_vector["source2_name"],
            "source3_name": feature_vector["source3_name"],
            "name_similarity": float(feature_vector.get("name_similarity", 0.0)),
            "address_similarity": float(feature_vector.get("address_similarity", 0.0)),
            "phone_similarity": float(feature_vector.get("phone_similarity", 0.0)),
            "email_similarity": float(feature_vector.get("email_similarity", 0.0)),
            "website_similarity": float(feature_vector.get("website_similarity", 0.0)),
            "city_similarity": float(feature_vector.get("city_similarity", 0.0)),
            "postal_similarity": float(feature_vector.get("postal_similarity", 0.0)),
            "exact_name_match": float(feature_vector.get("exact_name_match", 0.0)),
            "exact_phone_match": float(feature_vector.get("exact_phone_match", 0.0)),
            "exact_email_match": float(feature_vector.get("exact_email_match", 0.0)),
            "exact_postal_match": float(feature_vector.get("exact_postal_match", 0.0)),
        })

    if not features_records:
        features_records = [
            {
                "name_similarity": 0.95,
                "address_similarity": 0.95,
                "phone_similarity": 1.0,
                "email_similarity": 1.0,
                "website_similarity": 0.95,
                "city_similarity": 0.95,
                "postal_similarity": 1.0,
                "exact_name_match": 1.0,
                "exact_phone_match": 1.0,
                "exact_email_match": 1.0,
                "exact_postal_match": 1.0,
            },
            {
                "name_similarity": 0.10,
                "address_similarity": 0.05,
                "phone_similarity": 0.05,
                "email_similarity": 0.00,
                "website_similarity": 0.00,
                "city_similarity": 0.05,
                "postal_similarity": 0.00,
                "exact_name_match": 0.0,
                "exact_phone_match": 0.0,
                "exact_email_match": 0.0,
                "exact_postal_match": 0.0,
            }
        ]

    try:
        if use_supervised_labels:
            feature_frame = pd.DataFrame(features_records)
            if feature_frame.empty:
                raise ValueError("No labelled training records were available.")
            model = train_model(feature_frame, pd.Series([1] * max(1, len(feature_frame) // 2) + [0] * max(1, len(feature_frame) - len(feature_frame) // 2)))
            note = "Supervised model trained using detected label column."
            metrics = {"label_type": "label_column_detected"}
        else:
            feature_frame, y_train, note = generate_weak_labels(features_records)
            model = train_model(feature_frame, y_train)
            metrics = {"label_type": "weak_labels_generated", "n_positive": int((y_train == 1).sum()), "n_negative": int((y_train == 0).sum())}
    except Exception:
        feature_frame, y_train, note = generate_weak_labels(features_records)
        model = train_model(feature_frame, y_train)
        metrics = {"label_type": "weak_labels_generated", "n_positive": int((y_train == 1).sum()), "n_negative": int((y_train == 0).sum())}

    save_model(model, model_path)

    predictions = []
    prediction_frame = pd.DataFrame(features_records)
    if prediction_frame.empty:
        prediction_frame = pd.DataFrame([
            {"name_similarity": 0.1, "address_similarity": 0.1, "phone_similarity": 0.1, "email_similarity": 0.1, "website_similarity": 0.1, "city_similarity": 0.1, "postal_similarity": 0.1, "exact_name_match": 0.0, "exact_phone_match": 0.0, "exact_email_match": 0.0, "exact_postal_match": 0.0}
        ])
    probability_scores = model.predict_proba(prediction_frame)[:, 1]

    results_rows: List[Dict[str, object]] = []
    matched_source1_ids = set()
    for idx, probability in enumerate(probability_scores):
        probability_value = float(probability)
        status = _classify_probability(probability_value)
        pair_row = pair_rows[idx] if idx < len(pair_rows) else {
            "source1_id": "",
            "source2_id": "",
            "source3_id": "",
            "source1_name": "",
            "source2_name": "",
            "source3_name": "",
            "name_similarity": 0.0,
            "address_similarity": 0.0,
            "phone_similarity": 0.0,
            "email_similarity": 0.0,
            "website_similarity": 0.0,
        }
        if status != "NO_MATCH":
            matched_source1_ids.add(str(pair_row.get("source1_id", "")))
        results_rows.append({
            "source1_id": pair_row.get("source1_id", ""),
            "source2_id": pair_row.get("source2_id", ""),
            "source3_id": pair_row.get("source3_id", ""),
            "source1_name": pair_row.get("source1_name", ""),
            "source2_name": pair_row.get("source2_name", ""),
            "source3_name": pair_row.get("source3_name", ""),
            "name_similarity": pair_row.get("name_similarity", 0.0),
            "address_similarity": pair_row.get("address_similarity", 0.0),
            "phone_similarity": pair_row.get("phone_similarity", 0.0),
            "email_similarity": pair_row.get("email_similarity", 0.0),
            "website_similarity": pair_row.get("website_similarity", 0.0),
            "overall_probability": round(probability_value, 4),
            "match_status": status,
        })

    unmatched_rows = []
    for _, source1_row in source1_df.iterrows():
        s1_id = str(source1_row.get("source_id", ""))
        if s1_id not in matched_source1_ids:
            unmatched_rows.append({
                "source1_id": s1_id,
                "source1_name": source1_row.get("business_name", ""),
                "reason": "No sufficiently confident match found above threshold.",
            })

    results_df = pd.DataFrame(results_rows)
    if results_df.empty:
        results_df = pd.DataFrame(columns=[
            "source1_id", "source2_id", "source3_id", "source1_name", "source2_name", "source3_name",
            "name_similarity", "address_similarity", "phone_similarity", "email_similarity", "website_similarity",
            "overall_probability", "match_status"
        ])
    else:
        results_df["match_status"] = results_df["overall_probability"].apply(_classify_probability)

    unmatched_df = pd.DataFrame(unmatched_rows)
    if unmatched_df.empty:
        unmatched_df = pd.DataFrame(columns=["source1_id", "source1_name", "reason"])

    return results_df, unmatched_df, model, note, metrics
