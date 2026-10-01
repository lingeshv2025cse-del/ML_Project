from __future__ import annotations

from rapidfuzz import fuzz


def safe_value(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_for_fuzzy(value):
    return safe_value(value).lower()


def ratio_similarity(value1, value2):
    first = normalize_for_fuzzy(value1)
    second = normalize_for_fuzzy(value2)
    if not first or not second:
        return 0.0
    return fuzz.ratio(first, second) / 100.0


def wratio_similarity(value1, value2):
    first = normalize_for_fuzzy(value1)
    second = normalize_for_fuzzy(value2)
    if not first or not second:
        return 0.0
    return fuzz.WRatio(first, second) / 100.0


def token_sort_similarity(value1, value2):
    first = normalize_for_fuzzy(value1)
    second = normalize_for_fuzzy(value2)
    if not first or not second:
        return 0.0
    return fuzz.token_sort_ratio(first, second) / 100.0


def token_set_similarity(value1, value2):
    first = normalize_for_fuzzy(value1)
    second = normalize_for_fuzzy(value2)
    if not first or not second:
        return 0.0
    return fuzz.token_set_ratio(first, second) / 100.0


def company_name_similarity(name1, name2):
    score = (
        ratio_similarity(name1, name2)
        + wratio_similarity(name1, name2)
        + token_sort_similarity(name1, name2)
        + token_set_similarity(name1, name2)
    ) / 4.0
    return round(score, 4)


def address_similarity(address1, address2):
    return round(ratio_similarity(address1, address2), 4)


def city_similarity(city1, city2):
    return round(ratio_similarity(city1, city2), 4)


def postal_similarity(postal1, postal2):
    left = safe_value(postal1)
    right = safe_value(postal2)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return round(fuzz.ratio(left, right) / 100.0, 4)


def phone_similarity(phone1, phone2):
    left = safe_value(phone1).replace(" ", "")
    right = safe_value(phone2).replace(" ", "")
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left.endswith(right) or right.endswith(left):
        return 0.9
    common_suffix = 4
    left_suffix = left[-common_suffix:] if len(left) >= common_suffix else left
    right_suffix = right[-common_suffix:] if len(right) >= common_suffix else right
    if left_suffix == right_suffix:
        return 0.75
    return round(fuzz.ratio(left, right) / 100.0, 4)


def email_similarity(email1, email2):
    left = safe_value(email1).lower().strip()
    right = safe_value(email2).lower().strip()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_local = left.split("@")[-2] if "@" in left else left
    right_local = right.split("@")[-2] if "@" in right else right
    if left.split("@")[-1] == right.split("@")[-1]:
        return round((0.6 + 0.4 * (fuzz.ratio(left_local, right_local) / 100.0)), 4)
    return 0.0


def website_similarity(website1, website2):
    left = safe_value(website1).lower().replace("http://", "").replace("https://", "").replace("www.", "")
    right = safe_value(website2).lower().replace("http://", "").replace("https://", "").replace("www.", "")
    left = left.rstrip("/")
    right = right.rstrip("/")
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_domain = left.split("/")[0].split(":")[-1]
    right_domain = right.split("/")[0].split(":")[-1]
    if left_domain == right_domain:
        return 0.9
    return round(fuzz.ratio(left, right) / 100.0, 4)
