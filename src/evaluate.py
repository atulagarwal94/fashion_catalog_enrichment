from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FashionCatalogDataset, build_transforms
from src.model import build_model_from_label_info
from src.tasks import get_tasks
from src.utils import ensure_dir, get_device, load_config, save_json


def _split_suffix(split: str, tag: str | None) -> str:
    return f"{split}_{tag}" if tag else split


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label_info = checkpoint["label_info"]
    config = checkpoint["config"]
    model = build_model_from_label_info(label_info, checkpoint["backbone"], config["training"]["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def plot_confusion_matrix(cm, labels, title, output_path):
    fig = plt.figure(figsize=(max(8, len(labels) * 0.65), max(6, len(labels) * 0.55)))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(args):
    config = load_config(args.config)
    metrics_dir = ensure_dir(config["paths"]["metrics_dir"])
    split_csv = Path(config["paths"]["processed_dir"]) / f"{args.split}.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Could not find split CSV: {split_csv}")

    device = get_device()
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    label_info = checkpoint["label_info"]
    tasks = get_tasks(label_info)
    image_size = checkpoint.get("image_size", config["training"]["image_size"])
    dataset = FashionCatalogDataset(split_csv, transform=build_transforms(image_size, train=False))
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"], pin_memory=torch.cuda.is_available())

    all_true = {task: [] for task in tasks}
    all_pred = {task: [] for task in tasks}
    prediction_rows = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {args.split}"):
            images = batch["image"].to(device)
            outputs = model(images)
            batch_preds = {}
            batch_truth = {}
            for task in tasks:
                batch_preds[task] = outputs[task].argmax(dim=1).detach().cpu().numpy().tolist()
                batch_truth[task] = batch["targets"][task].numpy().tolist()
                all_pred[task].extend(batch_preds[task])
                all_true[task].extend(batch_truth[task])
            for i, image_path in enumerate(batch["image_path"]):
                row = {"image_path": image_path}
                for task in tasks:
                    row[f"true_{task}_idx"] = batch_truth[task][i]
                    row[f"pred_{task}_idx"] = batch_preds[task][i]
                prediction_rows.append(row)

    tag = getattr(args, "tag", None)
    backbone = getattr(args, "backbone", None) or checkpoint.get("backbone")
    suffix = _split_suffix(args.split, tag)

    metrics = {
        "split": args.split,
        "num_samples": len(dataset),
        "checkpoint": str(args.checkpoint),
        "backbone": backbone,
        "tag": tag,
    }
    for task in tasks:
        labels = [label_info[task]["idx_to_label"][str(idx)] for idx in range(label_info[task]["num_classes"])]
        metrics[f"{task}_accuracy"] = accuracy_score(all_true[task], all_pred[task])
        metrics[f"{task}_macro_f1"] = f1_score(all_true[task], all_pred[task], average="macro", zero_division=0)
        metrics[f"{task}_weighted_f1"] = f1_score(all_true[task], all_pred[task], average="weighted", zero_division=0)
        report = classification_report(all_true[task], all_pred[task], target_names=labels, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(metrics_dir / f"classification_report_{task}_{suffix}.csv")
        cm = confusion_matrix(all_true[task], all_pred[task], labels=list(range(len(labels))))
        title = f"{task.title()} Confusion Matrix - {suffix}"
        plot_confusion_matrix(cm, labels, title, metrics_dir / f"confusion_matrix_{task}_{suffix}.png")

    metrics["joint_accuracy"] = float(np.mean([all(all_true[task][i] == all_pred[task][i] for task in tasks) for i in range(len(all_true["class"]))]))
    save_json(metrics, metrics_dir / f"evaluation_{suffix}.json")
    pd.DataFrame(prediction_rows).to_csv(metrics_dir / f"predictions_{suffix}.csv", index=False)
    print(metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/models/best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix for output files, e.g. efficientnet_b0 → evaluation_test_efficientnet_b0.json",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
