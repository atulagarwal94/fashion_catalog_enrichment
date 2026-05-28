from __future__ import annotations
from typing import Dict, Optional


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_reverse_article_type_mapping(selected_article_types: Dict[str, list[str]]) -> Dict[str, str]:
    reverse_map: Dict[str, str] = {}
    for final_class, raw_values in selected_article_types.items():
        for raw_value in raw_values:
            reverse_map[normalize_text(raw_value).lower()] = final_class
    return reverse_map


def map_article_type(raw_article_type: object, selected_article_types: Dict[str, list[str]]) -> Optional[str]:
    reverse_map = build_reverse_article_type_mapping(selected_article_types)
    return reverse_map.get(normalize_text(raw_article_type).lower())


def map_gender(raw_gender: object, gender_mapping: Dict[str, str]) -> Optional[str]:
    gender = normalize_text(raw_gender)
    if not gender:
        return None
    return gender_mapping.get(gender)


def map_usage(raw_usage: object, usage_mapping: Dict[str, str], rare_usage_label: str = "Other") -> str:
    usage = normalize_text(raw_usage)
    if not usage:
        return rare_usage_label
    if usage in usage_mapping:
        return usage_mapping[usage]
    lowered = {normalize_text(k).lower(): v for k, v in usage_mapping.items()}
    return lowered.get(usage.lower(), rare_usage_label)


def map_color(raw_color: object, keep_list: list[str], rare_color_label: str = "Other") -> Optional[str]:
    color = normalize_text(raw_color)
    if not color:
        return None
    keep_set = {item.lower(): item for item in keep_list}
    return keep_set.get(color.lower(), rare_color_label)


def make_label_maps(labels: list[str]) -> dict:
    labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {str(idx): label for label, idx in label_to_idx.items()}
    return {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label, "num_classes": len(labels)}
