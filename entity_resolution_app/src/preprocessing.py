from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

COMMON_ALIASES = {
    "source_id": ["source_id", "id", "record_id", "entity_id", "business_id", "company_id", "customer_id"],
    "business_name": ["business_name", "name", "company_name", "company", "organization", "entity_name", "business"],
    "address": ["address", "street_address", "street", "location", "addr", "mailing_address", "address_line_1"],
    "city": ["city", "town", "locality"],
    "state": ["state", "province", "region"],
    "postal_code": ["postal_code", "zip", "zipcode", "postcode", "postal", "zip_code"],
    "phone": ["phone", "telephone", "phone_number", "mobile", "contact_number", "tel", "phone_no"],
    "email": ["email", "email_address", "e_mail", "contact_email", "mail"],
    "website": ["website", "url", "web", "site", "homepage", "domain", "website_url"],
}


def normalize_unicode(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = normalize_unicode(str(value)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" pvt ", " private ")
    text = text.replace(" ltd ", " limited ")
    text = text.replace(" llc ", " limited liability company ")
    text = text.replace(" inc ", " incorporated ")
    text = text.replace(" co ", " company ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_phone(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    digits = re.sub(r"\D+", "", text)
    if not digits:
        return ""
    if len(digits) > 10 and digits.startswith("00"):
        digits = digits[2:]
    if len(digits) > 10 and digits.startswith("91"):
        digits = digits[2:]
    return digits[-10:] if len(digits) > 10 else digits


def normalize_email(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = normalize_unicode(text)
    return text.strip()


def normalize_website(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = normalize_unicode(text)
    text = text.replace("http://", "").replace("https://", "")
    text = text.replace("www.", "")
    text = text.rstrip("/")
    text = re.sub(r"[^a-z0-9./-]", "", text)
    return text


def normalize_city(value: object) -> str:
    return normalize_text(value)


def normalize_postal(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"\D+", "", text)


def read_csv_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            if df.empty:
                return df
            return df
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise ValueError(f"Unable to read CSV file {file_path}: {last_error}")


def normalize_column_name(column_name: object) -> str:
    return normalize_text(str(column_name))


def detect_column_mapping(columns: Iterable[str]) -> Dict[str, str]:
    normalized_lookup: Dict[str, str] = {}
    for column in columns:
        normalized_lookup[normalize_column_name(column)] = str(column)

    mapping: Dict[str, str] = {}
    for canonical_name, aliases in COMMON_ALIASES.items():
        for alias in aliases:
            alias_key = normalize_column_name(alias)
            if alias_key in normalized_lookup:
                mapping[canonical_name] = normalized_lookup[alias_key]
                break
    return mapping


def prepare_source_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df is None:
        raise ValueError(f"{source_name} data is missing or invalid.")

    if df.empty:
        return pd.DataFrame({
            "source_id": [],
            "business_name": [],
            "address": [],
            "city": [],
            "state": [],
            "postal_code": [],
            "phone": [],
            "email": [],
            "website": [],
            "source_name": [],
        })

    working = df.copy()
    working.columns = [str(col).strip() for col in working.columns]
    mapping = detect_column_mapping(working.columns)

    for canonical, col_name in mapping.items():
        working[canonical] = working[col_name]

    for canonical_name in ["source_id", "business_name", "address", "city", "state", "postal_code", "phone", "email", "website"]:
        if canonical_name not in working.columns:
            working[canonical_name] = ""

    working["source_id"] = working["source_id"].fillna("")
    if working["source_id"].astype(str).str.len().eq(0).all():
        working["source_id"] = [f"{source_name.upper()}_{idx + 1}" for idx in range(len(working))]
    else:
        working["source_id"] = working["source_id"].astype(str)

    working["business_name"] = working["business_name"].fillna("").apply(normalize_text)
    working["address"] = working["address"].fillna("").apply(normalize_text)
    working["city"] = working["city"].fillna("").apply(normalize_city)
    working["state"] = working["state"].fillna("").apply(normalize_text)
    working["postal_code"] = working["postal_code"].fillna("").apply(normalize_postal)
    working["phone"] = working["phone"].fillna("").apply(normalize_phone)
    working["email"] = working["email"].fillna("").apply(normalize_email)
    working["website"] = working["website"].fillna("").apply(normalize_website)
    working["source_name"] = source_name

    return working[["source_id", "business_name", "address", "city", "state", "postal_code", "phone", "email", "website", "source_name"]].copy()
