from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from PIL import Image

from src.inference import FashionCatalogPredictor
from src.utils import ensure_dir, load_config, save_json


def benchmark(args):
    config = load_config(args.config)
    metrics_dir = ensure_dir(config["paths"]["metrics_dir"])
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Could not find image_dir: {image_dir}")
    image_paths = []
    for extension in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(image_dir.glob(extension))
    image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    predictor = FashionCatalogPredictor(args.checkpoint)
    with Image.open(image_paths[0]) as image:
        predictor.predict_pil(image, image_name=image_paths[0].name)

    per_image_tat = []
    predictions = []
    for path in image_paths:
        result = predictor.predict_image_path(path)
        predictions.append(result)
        per_image_tat.append(result["tat_ms"])

    batch_paths = image_paths[: args.batch_size]
    batch_images = []
    batch_names = []
    for path in batch_paths:
        with Image.open(path) as image:
            batch_images.append(image.convert("RGB"))
            batch_names.append(path.name)

    batch_predictions = predictor.predict_batch_pil(batch_images, batch_names)
    batch_total_ms = batch_predictions[0]["batch_tat_ms"] if batch_predictions else 0.0
    batch_avg_ms = batch_predictions[0]["tat_ms"] if batch_predictions else 0.0

    latency_report = {
        "checkpoint": args.checkpoint,
        "image_dir": str(image_dir),
        "num_images": len(image_paths),
        "single_image_latency_ms": {
            "mean": round(statistics.mean(per_image_tat), 3),
            "median": round(statistics.median(per_image_tat), 3),
            "min": round(min(per_image_tat), 3),
            "max": round(max(per_image_tat), 3),
            "p95": round(sorted(per_image_tat)[int(0.95 * (len(per_image_tat) - 1))], 3),
        },
        "true_batch_inference": True,
        "batch_size": len(batch_predictions),
        "batch_forward_ms": round(batch_total_ms, 3),
        "batch_avg_ms_per_image": round(batch_avg_ms, 3),
        "speedup_vs_single_mean": round(statistics.mean(per_image_tat) / batch_avg_ms, 3) if batch_avg_ms else None,
    }
    save_json(latency_report, metrics_dir / "latency_benchmark.json")
    save_json({"predictions": predictions}, metrics_dir / "sample_predictions.json")
    print(latency_report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pt")
    parser.add_argument("--image_dir", default="data/sample_images")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=100)
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
