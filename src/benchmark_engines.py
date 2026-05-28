from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FashionCatalogDataset, build_transforms
from src.model import build_model_from_label_info
from src.tasks import get_tasks
from src.utils import ensure_dir, get_device, load_config, save_json


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    return float(vs[int(0.95 * (len(vs) - 1))])


def _summarize_ms_per_image(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "mean": round(float(statistics.mean(values)), 3),
        "median": round(float(statistics.median(values)), 3),
        "p95": round(_p95(values), 3),
        "min": round(float(min(values)), 3),
        "max": round(float(max(values)), 3),
        "num_batches": int(len(values)),
    }


def _load_torch_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_info = checkpoint["label_info"]
    config = checkpoint["config"]
    tasks = get_tasks(label_info)
    model = build_model_from_label_info(label_info, checkpoint["backbone"], config["training"]["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, tasks


def _load_ort_session(onnx_path: Path, providers: list[str] | None = None) -> tuple[ort.InferenceSession, str, list]:
    providers = providers or ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    inp = session.get_inputs()[0]
    return session, inp.name, list(inp.shape)


def benchmark(args, output_path: str | Path | None = None) -> dict:
    config = load_config(args.config)
    metrics_dir = ensure_dir(config["paths"]["metrics_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    split_csv = processed_dir / f"{args.split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Could not find split CSV: {split_csv}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {ckpt_path}")

    onnx_path = Path(args.onnx_model)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Could not find ONNX model: {onnx_path}\n"
            "Create it with:\n"
            f"  python -m src.optimize_onnx --checkpoint {args.checkpoint} --output_dir outputs/models"
        )

    device = get_device()
    torch_model, checkpoint, tasks = _load_torch_model(ckpt_path, device)
    image_size = checkpoint.get("image_size", config["training"]["image_size"])

    dataset = FashionCatalogDataset(split_csv, transform=build_transforms(image_size, train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    ort_session, ort_input_name, ort_input_shape = _load_ort_session(onnx_path)
    output_names = [f"{t}_logits" for t in tasks]

    static_batch_1 = isinstance(ort_input_shape[0], int) and ort_input_shape[0] == 1

    # Warmup
    warmup_batches = 0
    for batch in loader:
        images = batch["image"]
        _ = torch_model(images.to(device))
        images_np = images.numpy().astype(np.float32, copy=False)
        if static_batch_1 and images_np.shape[0] != 1:
            for i in range(images_np.shape[0]):
                _ = ort_session.run(output_names, {ort_input_name: images_np[i : i + 1]})
        else:
            _ = ort_session.run(output_names, {ort_input_name: images_np})
        warmup_batches += 1
        if warmup_batches >= args.warmup_batches:
            break

    # Benchmark
    torch_ms_per_image: list[float] = []
    ort_ms_per_image: list[float] = []

    batches_seen = 0
    for batch in tqdm(loader, desc=f"Benchmarking engines on {args.split}"):
        images = batch["image"]
        bsz = int(images.size(0))

        # PyTorch forward
        start = time.perf_counter()
        with torch.no_grad():
            _ = torch_model(images.to(device))
        torch_ms_per_image.append(((time.perf_counter() - start) * 1000) / max(bsz, 1))

        # ORT forward (float32 input)
        images_np = images.numpy().astype(np.float32, copy=False)
        start = time.perf_counter()
        if static_batch_1 and bsz != 1:
            for i in range(bsz):
                _ = ort_session.run(output_names, {ort_input_name: images_np[i : i + 1]})
        else:
            _ = ort_session.run(output_names, {ort_input_name: images_np})
        ort_ms_per_image.append(((time.perf_counter() - start) * 1000) / max(bsz, 1))

        batches_seen += 1
        if args.max_batches and batches_seen >= args.max_batches:
            break

    report = {
        "split": args.split,
        "num_samples": int(len(dataset)),
        "batch_size": int(args.batch_size),
        "max_batches": int(args.max_batches) if args.max_batches else None,
        "checkpoint": args.checkpoint,
        "backbone": checkpoint.get("backbone"),
        "onnx_model": str(onnx_path),
        "onnx_providers": ort_session.get_providers(),
        "onnx_input_shape": ort_input_shape,
        "onnx_static_batch_1": bool(static_batch_1),
        "tasks": tasks,
        "pytorch_tat_ms_per_image": _summarize_ms_per_image(torch_ms_per_image),
        "onnxruntime_tat_ms_per_image": _summarize_ms_per_image(ort_ms_per_image),
        "notes": "Forward-pass time only. DataLoader + transforms excluded as much as possible.",
    }

    out_path = Path(output_path) if output_path else (Path(metrics_dir) / f"engine_benchmark_{args.split}.json")
    save_json(report, out_path)
    print(report)
    print(f"Saved: {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime on a dataset split.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pt")
    parser.add_argument("--onnx-model", default="outputs/models/model_quantized.onnx")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50, help="Limit number of batches to benchmark (0 = full split)")
    args = parser.parse_args()

    if args.max_batches == 0:
        args.max_batches = None

    benchmark(args)


if __name__ == "__main__":
    main()

