from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from src.dataset import build_transforms
from src.tasks import get_tasks
from src.utils import get_device


class ONNXFashionCatalogPredictor:
    """ONNX Runtime predictor using label_info from a PyTorch checkpoint.

    This keeps the label maps consistent with training, but runs the model forward
    using ONNX Runtime (optionally quantized INT8 model).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        onnx_path: str | Path,
        providers: list[str] | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Could not find checkpoint: {self.checkpoint_path}")

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"Could not find ONNX model: {self.onnx_path}")

        # Load label maps + config from checkpoint (CPU-safe)
        device = get_device()
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        self.label_info = checkpoint["label_info"]
        self.tasks = get_tasks(self.label_info)
        self.config = checkpoint["config"]
        self.backbone = checkpoint["backbone"]
        self.image_size = checkpoint.get("image_size", self.config["training"]["image_size"])
        self.transform = build_transforms(self.image_size, train=False)

        # ORT session (CPU by default)
        self.providers = providers or ["CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=sess_opts, providers=self.providers)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = list(self.session.get_inputs()[0].shape)
        self.output_names = [o.name for o in self.session.get_outputs()]

        expected = [f"{t}_logits" for t in self.tasks]
        missing = [n for n in expected if n not in self.output_names]
        if missing:
            raise ValueError(
                "ONNX outputs do not match tasks from checkpoint.\n"
                f"Tasks: {self.tasks}\n"
                f"Expected outputs: {expected}\n"
                f"Found outputs: {self.output_names}\n"
                f"Missing: {missing}\n"
                "Re-export ONNX for the same checkpoint via `python -m src.optimize_onnx`."
            )

    def _decode_prediction(self, task: str, logits_1d: np.ndarray, top_k: int = 3) -> Dict:
        # stable softmax
        x = logits_1d.astype(np.float64)
        x = x - np.max(x)
        exp = np.exp(x)
        probs = exp / np.sum(exp)

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        k = min(int(top_k), int(probs.shape[-1]))
        top_idx = np.argsort(-probs)[:k]

        return {
            "label": self.label_info[task]["idx_to_label"][str(idx)],
            "confidence": round(conf, 4),
            "top_predictions": [
                {
                    "label": self.label_info[task]["idx_to_label"][str(int(i))],
                    "confidence": round(float(probs[int(i)]), 4),
                }
                for i in top_idx
            ],
        }

    def predict_batch_pil(self, images: List[Image.Image], image_names: List[str] | None = None) -> List[Dict]:
        if not images:
            return []
        names = image_names or [f"image_{i}" for i in range(len(images))]
        if len(names) != len(images):
            raise ValueError("image_names length must match images length")

        # Transform via torchvision → torch tensor → numpy for ORT
        batch_t = torch.stack([self.transform(img.convert("RGB")) for img in images])
        batch_np = batch_t.detach().cpu().numpy().astype(np.float32, copy=False)

        # Some exported/quantized ONNX graphs are static batch=1.
        # If so, fall back to running per-image and aggregate.
        expected_b0 = self.input_shape[0]
        wants_batch_1 = isinstance(expected_b0, int) and expected_b0 == 1

        start = time.perf_counter()
        if wants_batch_1 and batch_np.shape[0] != 1:
            per_image_outputs = []
            for i in range(batch_np.shape[0]):
                out_i = self.session.run([f"{t}_logits" for t in self.tasks], {self.input_name: batch_np[i : i + 1]})
                per_image_outputs.append(out_i)
            # stitch back into list-of-[N,C]
            ort_outputs = []
            for task_idx in range(len(self.tasks)):
                ort_outputs.append(np.concatenate([o[task_idx] for o in per_image_outputs], axis=0))
        else:
            ort_outputs = self.session.run([f"{t}_logits" for t in self.tasks], {self.input_name: batch_np})

        batch_tat_ms = (time.perf_counter() - start) * 1000
        avg_tat_ms = batch_tat_ms / len(images)

        # ort_outputs is list aligned to tasks; each is [N, C]
        results: list[Dict] = []
        for row_idx, name in enumerate(names):
            prediction = {}
            for task, logits in zip(self.tasks, ort_outputs):
                prediction[task] = self._decode_prediction(task, logits[row_idx])
            overall_confidence = min(prediction[t]["confidence"] for t in self.tasks)
            results.append(
                {
                    "image_name": name,
                    "prediction": prediction,
                    "overall_confidence": round(float(overall_confidence), 4),
                    "tat_ms": round(float(avg_tat_ms), 3),
                    "batch_tat_ms": round(float(batch_tat_ms), 3),
                    "batch_size": len(images),
                    "engine": "onnxruntime",
                    "onnx_path": str(self.onnx_path),
                    "providers": list(self.providers),
                }
            )
        return results

    def predict_pil(self, image: Image.Image, image_name: str = "uploaded_image") -> Dict:
        return self.predict_batch_pil([image], [image_name])[0]

