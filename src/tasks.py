"""Shared task names for multi-head training and inference."""

from __future__ import annotations

from typing import Dict, List

CORE_TASKS = ["class", "gender", "color"]
OPTIONAL_TASKS = ["usage"]


def get_tasks(label_info: Dict | None = None) -> List[str]:
    tasks = list(CORE_TASKS)
    if label_info and "usage" in label_info:
        tasks.append("usage")
    return tasks
