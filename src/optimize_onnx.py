from __future__ import annotations

import argparse
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.model import ONNXExportWrapper, build_model_from_label_info
from src.tasks import get_tasks
from src.utils import ensure_dir, get_device, load_config, model_file_size_mb, save_json


def export_and_quantize(args):
    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    label_info = checkpoint["label_info"]
    tasks = get_tasks(label_info)
    image_size = checkpoint.get("image_size", config["training"]["image_size"])

    model = build_model_from_label_info(label_info, checkpoint["backbone"], config["training"]["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    wrapper = ONNXExportWrapper(model).to(device)
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    onnx_path = Path(output_dir) / "model.onnx"
    quantized_path = Path(output_dir) / "model_quantized.onnx"

    output_names = [f"{task}_logits" for task in tasks]
    dynamic_axes = {"image": {0: "batch_size"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=config["optimization"]["onnx_opset"],
        do_constant_folding=True,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    quantization_status = "skipped"
    if config["optimization"].get("quantize_dynamic", True):
        quantize_dynamic(model_input=str(onnx_path), model_output=str(quantized_path), weight_type=QuantType.QInt8)
        quantization_status = "completed"

    report = {
        "checkpoint": str(args.checkpoint),
        "onnx_path": str(onnx_path),
        "quantized_path": str(quantized_path) if quantized_path.exists() else None,
        "onnx_size_mb": model_file_size_mb(onnx_path),
        "quantized_size_mb": model_file_size_mb(quantized_path) if quantized_path.exists() else None,
        "quantization_status": quantization_status,
        "output_names": output_names,
    }
    save_json(report, Path(config["paths"]["metrics_dir"]) / "onnx_optimization_report.json")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pt")
    parser.add_argument("--output_dir", default="outputs/models")
    args = parser.parse_args()
    export_and_quantize(args)


if __name__ == "__main__":
    main()
