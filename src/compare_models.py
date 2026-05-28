"""Evaluate all backbone checkpoints and print a side-by-side comparison table.

Usage:
    python -m src.compare_models
    python -m src.compare_models --split test --skip-eval
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.evaluate import _split_suffix, evaluate
from src.train_all import BACKBONES, checkpoint_path
from src.utils import load_config, load_json, save_json

COMPARE_COLUMNS = [
    "backbone",
    "class_accuracy",
    "gender_accuracy",
    "color_accuracy",
    "usage_accuracy",
    "tat_ms_mean",
    "joint_accuracy",
    "class_macro_f1",
    "gender_macro_f1",
    "color_macro_f1",
    "usage_macro_f1",
    "best_val_macro_f1",
    "checkpoint",
]


def _training_best_val(metrics_dir: Path, backbone: str) -> float | None:
    history_path = metrics_dir / f"training_history_{backbone}.json"
    if not history_path.exists():
        return None
    data = load_json(history_path)
    return data.get("best_val_macro_f1")


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def compare_models(args) -> pd.DataFrame:
    config = load_config(args.config)
    model_dir = Path(config["paths"]["model_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for backbone in args.backbones:
        ckpt = checkpoint_path(model_dir, backbone)
        suffix = _split_suffix(args.split, backbone)
        eval_path = metrics_dir / f"evaluation_{suffix}.json"

        if not ckpt.exists():
            print(f"SKIP {backbone}: checkpoint not found ({ckpt})")
            continue

        if not args.skip_eval:
            print(f"\nEvaluating {backbone} …")
            evaluate(
                Namespace(
                    config=args.config,
                    checkpoint=str(ckpt),
                    split=args.split,
                    tag=backbone,
                )
            )
        elif not eval_path.exists():
            print(f"SKIP {backbone}: no metrics at {eval_path} (run without --skip-eval)")
            continue

        metrics = load_json(eval_path)
        rows.append(
            {
                "backbone": backbone,
                "class_accuracy": metrics.get("class_accuracy"),
                "gender_accuracy": metrics.get("gender_accuracy"),
                "color_accuracy": metrics.get("color_accuracy"),
                "usage_accuracy": metrics.get("usage_accuracy"),
                "tat_ms_mean": (metrics.get("tat_ms_per_image") or {}).get("mean"),
                "joint_accuracy": metrics.get("joint_accuracy"),
                "class_macro_f1": metrics.get("class_macro_f1"),
                "gender_macro_f1": metrics.get("gender_macro_f1"),
                "color_macro_f1": metrics.get("color_macro_f1"),
                "usage_macro_f1": metrics.get("usage_macro_f1"),
                "best_val_macro_f1": _training_best_val(metrics_dir, backbone),
                "checkpoint": str(ckpt),
            }
        )

    if not rows:
        raise SystemExit("No models to compare. Train checkpoints first or run evaluation.")

    df = pd.DataFrame(rows)
    out_csv = metrics_dir / f"model_comparison_{args.split}.csv"
    out_json = metrics_dir / f"model_comparison_{args.split}.json"
    df.to_csv(out_csv, index=False)
    save_json({"split": args.split, "models": rows}, out_json)

    print("\n" + "=" * 72)
    print(f"Model comparison ({args.split} set)")
    print("=" * 72)
    display = df.copy()
    for col in [
        "class_accuracy",
        "gender_accuracy",
        "color_accuracy",
        "usage_accuracy",
        "joint_accuracy",
        "class_macro_f1",
        "gender_macro_f1",
        "color_macro_f1",
        "usage_macro_f1",
        "best_val_macro_f1",
    ]:
        if col in display.columns:
            display[col] = display[col].map(_format_pct)
    if "tat_ms_mean" in display.columns:
        display["tat_ms_mean"] = display["tat_ms_mean"].map(lambda v: "—" if v is None else f"{v:.1f} ms")
    print(display[COMPARE_COLUMNS].to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare test metrics across all backbones.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=BACKBONES,
        default=BACKBONES,
        help="Backbones to include (default: all three)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only aggregate existing evaluation_<split>_<backbone>.json files",
    )
    args = parser.parse_args()
    compare_models(args)


if __name__ == "__main__":
    main()
