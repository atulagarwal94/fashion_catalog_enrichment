"""Train all backbone models sequentially from the terminal.

Usage:
    python -m src.train_all
    python -m src.train_all --config configs/config.yaml
    python -m src.train_all --skip-existing
    python -m src.train_all --backbones resnet50 mobilenet_v3_large
"""

from __future__ import annotations

import argparse
import time
from argparse import Namespace
from pathlib import Path

from src.train import train
from src.utils import load_config

BACKBONES = ["efficientnet_b0", "resnet50", "mobilenet_v3_large"]


def checkpoint_path(model_dir: Path, backbone: str) -> Path:
    return model_dir / f"best_model_{backbone}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all fashion catalog backbones in one run.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=BACKBONES,
        default=BACKBONES,
        help="Which backbones to train (default: all three)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip backbones that already have best_model_<backbone>.pt",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch without ImageNet pretrained weights",
    )
    parser.add_argument(
        "--set-default",
        metavar="BACKBONE",
        choices=BACKBONES,
        default=None,
        help="After training, copy this backbone's checkpoint to outputs/models/best_model.pt",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training backbones:", ", ".join(args.backbones))
    print("Config:", args.config)
    print("Model dir:", model_dir.resolve())
    print("=" * 60)

    completed = []
    skipped = []
    failed = []

    for i, backbone in enumerate(args.backbones, start=1):
        ckpt = checkpoint_path(model_dir, backbone)
        if args.skip_existing and ckpt.exists():
            print(f"\n[{i}/{len(args.backbones)}] SKIP {backbone} — already exists: {ckpt}")
            skipped.append(backbone)
            continue

        print(f"\n[{i}/{len(args.backbones)}] START {backbone}")
        print("-" * 60)
        started = time.perf_counter()
        try:
            train(
                Namespace(
                    config=args.config,
                    backbone=backbone,
                    no_pretrained=args.no_pretrained,
                )
            )
            elapsed_min = (time.perf_counter() - started) / 60
            print(f"Finished {backbone} in {elapsed_min:.1f} min → {ckpt}")
            completed.append((backbone, elapsed_min))
        except Exception as exc:
            print(f"FAILED {backbone}: {exc}")
            failed.append((backbone, str(exc)))

    if args.set_default:
        default_ckpt = checkpoint_path(model_dir, args.set_default)
        if default_ckpt.exists():
            import shutil

            target = model_dir / "best_model.pt"
            shutil.copy2(default_ckpt, target)
            print(f"\nDefault checkpoint set: {target} (from {args.set_default})")
        else:
            print(f"\nWarning: --set-default {args.set_default} but {default_ckpt} not found.")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if completed:
        print("Trained:")
        for backbone, mins in completed:
            print(f"  ✓ {backbone} ({mins:.1f} min)")
    if skipped:
        print("Skipped (already trained):")
        for backbone in skipped:
            print(f"  - {backbone}")
    if failed:
        print("Failed:")
        for backbone, err in failed:
            print(f"  ✗ {backbone}: {err}")
    print("\nCheckpoints:")
    for backbone in BACKBONES:
        ckpt = checkpoint_path(model_dir, backbone)
        status = "exists" if ckpt.exists() else "missing"
        print(f"  {backbone}: {ckpt} [{status}]")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
