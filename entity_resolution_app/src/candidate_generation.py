from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Set, Tuple

import pandas as pd

from config import TOP_K_CANDIDATES


def _block_values_for_record(row: pd.Series) -> List[str]:
    values: List[str] = []
    business_name = str(row.get("business_name", "") or "").strip()
    city = str(row.get("city", "") or "").strip()
    postal = str(row.get("postal_code", "") or "").strip()
    phone = str(row.get("phone", "") or "").strip()
    email = str(row.get("email", "") or "").strip()
    website = str(row.get("website", "") or "").strip()

    if len(business_name) >= 3:
        values.append(business_name[:3])
        values.append(business_name.split()[0][:3] if business_name.split() else business_name[:3])
    if city:
        values.append(city)
    if postal:
        values.append(postal)
    if phone:
        values.append(phone[-4:])
    if email and "@" in email:
        values.append(email.split("@")[-1].split(".")[0])
    if website:
        domain = website.split("/")[0].split(":")[-1]
        if "." in domain:
            values.append(domain.split(".")[-2])
    return [value for value in values if value]


def _build_block_index(frame: pd.DataFrame) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for idx, row in frame.iterrows():
        for key in _block_values_for_record(row):
            index[key].append(str(row.get("source_id", idx)))
    return index


def generate_candidates(source1_df: pd.DataFrame, source2_df: pd.DataFrame, source3_df: pd.DataFrame, top_k: int = TOP_K_CANDIDATES):
    source2_index = _build_block_index(source2_df)
    source3_index = _build_block_index(source3_df)

    candidates: List[Dict[str, object]] = []

    def add_candidates(source_name: str, source_df: pd.DataFrame, index: Dict[str, List[str]], target_name: str):
        seen_pairs: Set[Tuple[str, str]] = set()
        for _, row in source1_df.iterrows():
            s1_id = str(row.get("source_id", ""))
            block_values = _block_values_for_record(row)
            candidate_ids: Set[str] = set()
            for block_key in block_values:
                candidate_ids.update(index.get(block_key, []))
            if not candidate_ids:
                continue
            limited_ids = list(candidate_ids)[:top_k]
            for candidate_id in limited_ids:
                pair_key = (s1_id, candidate_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                target_row = source_df[source_df["source_id"].astype(str) == str(candidate_id)]
                if target_row.empty:
                    continue
                match_row = target_row.iloc[0]
                candidates.append({
                    "source1_id": s1_id,
                    "source1_name": row.get("business_name", ""),
                    "source2_id": match_row.get("source_id", "") if source_name == "source2" else "",
                    "source3_id": match_row.get("source_id", "") if source_name == "source3" else "",
                    "candidate_id": candidate_id,
                    "candidate_name": match_row.get("business_name", ""),
                    "source_name": source_name,
                    "row_type": target_name,
                })

    add_candidates("source2", source2_df, source2_index, "source2")
    add_candidates("source3", source3_df, source3_index, "source3")

    return candidates
