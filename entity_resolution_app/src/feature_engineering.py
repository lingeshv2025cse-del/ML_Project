from __future__ import annotations

from typing import Dict

from src.similarity import (
    address_similarity,
    city_similarity,
    company_name_similarity,
    email_similarity,
    phone_similarity,
    postal_similarity,
    website_similarity,
)


FEATURE_COLUMNS = [
    "name_similarity",
    "address_similarity",
    "phone_similarity",
    "email_similarity",
    "website_similarity",
    "city_similarity",
    "postal_similarity",
    "exact_name_match",
    "exact_phone_match",
    "exact_email_match",
    "exact_postal_match",
]


def build_feature_vector(source1_row: Dict[str, object], candidate_row: Dict[str, object]) -> Dict[str, float]:
    source1_name = str(source1_row.get("business_name", "") or "")
    candidate_name = str(candidate_row.get("business_name", "") or "")
    source1_address = str(source1_row.get("address", "") or "")
    candidate_address = str(candidate_row.get("address", "") or "")
    source1_phone = str(source1_row.get("phone", "") or "")
    candidate_phone = str(candidate_row.get("phone", "") or "")
    source1_email = str(source1_row.get("email", "") or "")
    candidate_email = str(candidate_row.get("email", "") or "")
    source1_website = str(source1_row.get("website", "") or "")
    candidate_website = str(candidate_row.get("website", "") or "")
    source1_city = str(source1_row.get("city", "") or "")
    candidate_city = str(candidate_row.get("city", "") or "")
    source1_postal = str(source1_row.get("postal_code", "") or "")
    candidate_postal = str(candidate_row.get("postal_code", "") or "")

    features = {
        "name_similarity": company_name_similarity(source1_name, candidate_name),
        "address_similarity": address_similarity(source1_address, candidate_address),
        "phone_similarity": phone_similarity(source1_phone, candidate_phone),
        "email_similarity": email_similarity(source1_email, candidate_email),
        "website_similarity": website_similarity(source1_website, candidate_website),
        "city_similarity": city_similarity(source1_city, candidate_city),
        "postal_similarity": postal_similarity(source1_postal, candidate_postal),
        "exact_name_match": 1.0 if source1_name and source1_name == candidate_name else 0.0,
        "exact_phone_match": 1.0 if source1_phone and source1_phone == candidate_phone else 0.0,
        "exact_email_match": 1.0 if source1_email and source1_email == candidate_email else 0.0,
        "exact_postal_match": 1.0 if source1_postal and source1_postal == candidate_postal else 0.0,
    }

    return features
