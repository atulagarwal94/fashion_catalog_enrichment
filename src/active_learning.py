from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.tasks import CORE_TASKS
from src.utils import ensure_dir


def _tasks_for_prediction(prediction: Dict) -> List[str]:
    pred = prediction.get("prediction", {})
    tasks = list(CORE_TASKS)
    if "usage" in pred:
        tasks.append("usage")
    return tasks


def should_flag_for_review(prediction: Dict, threshold: float) -> bool:
    for task in _tasks_for_prediction(prediction):
        task_prediction = prediction.get("prediction", {}).get(task, {})
        if float(task_prediction.get("confidence", 0.0)) < threshold:
            return True
    return False


def build_review_row(prediction: Dict, threshold: float) -> Dict:
    tasks = _tasks_for_prediction(prediction)
    row = {
        "image_name": prediction.get("image_name"),
        "threshold": threshold,
        "tat_ms": prediction.get("tat_ms"),
        "needs_review": should_flag_for_review(prediction, threshold),
    }
    for task in tasks:
        task_prediction = prediction.get("prediction", {}).get(task, {})
        row[f"pred_{task}"] = task_prediction.get("label")
        row[f"{task}_confidence"] = task_prediction.get("confidence")
    row.update({"reviewed_class": "", "reviewed_gender": "", "reviewed_color": "", "reviewed_usage": "", "reviewer_notes": ""})
    return row


def create_review_queue(predictions: Iterable[Dict], output_csv: str | Path, threshold: float = 0.70) -> List[Dict]:
    rows = [build_review_row(prediction, threshold) for prediction in predictions if should_flag_for_review(prediction, threshold)]
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = [
            "image_name",
            "threshold",
            "tat_ms",
            "needs_review",
            "pred_class",
            "class_confidence",
            "pred_gender",
            "gender_confidence",
            "pred_color",
            "color_confidence",
            "pred_usage",
            "usage_confidence",
            "reviewed_class",
            "reviewed_gender",
            "reviewed_color",
            "reviewed_usage",
            "reviewer_notes",
        ]
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_json", required=True)
    parser.add_argument("--output_csv", default="outputs/review_queue/review_queue.csv")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    with open(args.predictions_json, "r", encoding="utf-8") as file:
        payload = json.load(file)
    predictions = payload["predictions"] if isinstance(payload, dict) and "predictions" in payload else payload if isinstance(payload, list) else [payload]
    rows = create_review_queue(predictions, args.output_csv, args.threshold)
    print(f"Created review queue with {len(rows)} rows: {args.output_csv}")


if __name__ == "__main__":
    main()
