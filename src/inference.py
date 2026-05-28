from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from src.dataset import build_transforms
from src.model import build_model_from_label_info
from src.tasks import get_tasks
from src.utils import get_device, save_json


class FashionCatalogPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Could not find checkpoint: {self.checkpoint_path}")
        self.device = torch.device(device) if device else get_device()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.label_info = checkpoint["label_info"]
        self.tasks = get_tasks(self.label_info)
        self.config = checkpoint["config"]
        self.backbone = checkpoint["backbone"]
        self.image_size = checkpoint.get("image_size", self.config["training"]["image_size"])
        self.model = build_model_from_label_info(self.label_info, self.backbone, self.config["training"]["dropout"], pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = build_transforms(self.image_size, train=False)
        self._warmup()

    def _warmup(self) -> None:
        """Run one silent forward pass so the first real prediction is not penalised by GPU spin-up."""
        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.no_grad():
            self.model(dummy)

    def _decode_prediction(self, task: str, logits: torch.Tensor, batch_idx: int = 0, top_k: int = 3) -> Dict:
        row_logits = logits[batch_idx] if logits.dim() == 2 else logits
        probabilities = torch.softmax(row_logits, dim=-1)
        confidence, pred_idx = probabilities.max(dim=-1)
        idx = int(pred_idx.item())
        result = {
            "label": self.label_info[task]["idx_to_label"][str(idx)],
            "confidence": round(float(confidence.item()), 4),
        }
        k = min(top_k, probabilities.shape[-1])
        topk_vals, topk_idx = probabilities.topk(k)
        result["top_predictions"] = [
            {
                "label": self.label_info[task]["idx_to_label"][str(int(i.item()))],
                "confidence": round(float(v.item()), 4),
            }
            for v, i in zip(topk_vals, topk_idx)
        ]
        return result

    def predict_batch_pil(self, images: List[Image.Image], image_names: List[str] | None = None) -> List[Dict]:
        """Run one forward pass for a stack of images shaped [N, 3, H, W]."""
        if not images:
            return []
        names = image_names or [f"image_{i}" for i in range(len(images))]
        if len(names) != len(images):
            raise ValueError("image_names length must match images length")

        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(batch)
        batch_tat_ms = (time.perf_counter() - start) * 1000
        avg_tat_ms = batch_tat_ms / len(images)

        results = []
        for i, name in enumerate(names):
            prediction = {task: self._decode_prediction(task, outputs[task], batch_idx=i) for task in self.tasks}
            overall_confidence = min(prediction[task]["confidence"] for task in self.tasks)
            results.append(
                {
                    "image_name": name,
                    "prediction": prediction,
                    "overall_confidence": round(overall_confidence, 4),
                    "tat_ms": round(avg_tat_ms, 3),
                    "batch_tat_ms": round(batch_tat_ms, 3),
                    "batch_size": len(images),
                }
            )
        return results

    def predict_pil(self, image: Image.Image, image_name: str = "uploaded_image") -> Dict:
        return self.predict_batch_pil([image], [image_name])[0]

    def predict_image_path(self, image_path: str | Path) -> Dict:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            return self.predict_pil(image, image_name=image_path.name)

    def predict_batch_paths(self, image_paths: List[str | Path]) -> List[Dict]:
        images: List[Image.Image] = []
        names: List[str] = []
        for path in image_paths:
            path = Path(path)
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
                names.append(path.name)
        return self.predict_batch_pil(images, names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    predictor = FashionCatalogPredictor(args.checkpoint)
    result = predictor.predict_image_path(args.image)
    print(json.dumps(result, indent=2))
    if args.output_json:
        save_json(result, args.output_json)


if __name__ == "__main__":
    main()
