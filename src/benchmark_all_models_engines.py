from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.benchmark_engines import benchmark
from src.train_all import BACKBONES, checkpoint_path
from src.utils import load_config, save_json


def _fmt(v: float | None) -> float | None:
    return None if v is None else float(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX across all backbones and write a single summary file.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50, help="Limit number of batches per model (0 = full split)")
    parser.add_argument(
        "--onnx-root",
        default="outputs/models/onnx",
        help="Folder containing per-backbone ONNX models at <onnx-root>/<backbone>/model.onnx",
    )
    args = parser.parse_args()

    if args.max_batches == 0:
        args.max_batches = None

    config = load_config(args.config)
    model_dir = Path(config["paths"]["model_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_model_reports: dict[str, dict] = {}

    for backbone in BACKBONES:
        ckpt = checkpoint_path(model_dir, backbone)
        if not ckpt.exists():
            print(f"SKIP {backbone}: missing checkpoint {ckpt}")
            continue

        onnx_path = Path(args.onnx_root) / backbone / "model.onnx"
        if not onnx_path.exists():
            print(f"SKIP {backbone}: missing ONNX {onnx_path}")
            continue

        out_path = metrics_dir / f"engine_benchmark_{args.split}_{backbone}.json"
        report = benchmark(
            Namespace(
                config=args.config,
                split=args.split,
                checkpoint=str(ckpt),
                onnx_model=str(onnx_path),
                batch_size=args.batch_size,
                warmup_batches=args.warmup_batches,
                max_batches=args.max_batches,
            ),
            output_path=out_path,
        )
        per_model_reports[backbone] = report

        pt_mean = (report.get("pytorch_tat_ms_per_image") or {}).get("mean")
        onnx_mean = (report.get("onnxruntime_tat_ms_per_image") or {}).get("mean")
        speedup = (float(pt_mean) / float(onnx_mean)) if (pt_mean and onnx_mean) else None

        rows.append(
            {
                "backbone": backbone,
                "split": args.split,
                "batch_size": args.batch_size,
                "onnx_model": str(onnx_path),
                "onnx_static_batch_1": report.get("onnx_static_batch_1"),
                "pytorch_mean_ms_per_image": _fmt(pt_mean),
                "onnx_mean_ms_per_image": _fmt(onnx_mean),
                "pytorch_p95_ms_per_image": _fmt((report.get("pytorch_tat_ms_per_image") or {}).get("p95")),
                "onnx_p95_ms_per_image": _fmt((report.get("onnxruntime_tat_ms_per_image") or {}).get("p95")),
                "speedup_pytorch_over_onnx": _fmt(speedup),
            }
        )

    if not rows:
        raise SystemExit(
            "No benchmarks were run. Ensure you have checkpoints at outputs/models/best_model_<backbone>.pt "
            "and ONNX models at outputs/models/onnx/<backbone>/model.onnx (export with --dynamic-batch --no-quantize)."
        )

    df = pd.DataFrame(rows).sort_values(["split", "backbone"]).reset_index(drop=True)
    out_csv = metrics_dir / f"engine_benchmark_summary_{args.split}.csv"
    out_json = metrics_dir / f"engine_benchmark_summary_{args.split}.json"
    df.to_csv(out_csv, index=False)
    save_json({"split": args.split, "rows": rows, "per_model_reports": per_model_reports}, out_json)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()

