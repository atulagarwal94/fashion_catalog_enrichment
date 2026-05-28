from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dataset import FashionCatalogDataset, build_transforms
from src.model import build_model_from_label_info
from src.tasks import get_tasks
from src.utils import count_trainable_parameters, ensure_dir, get_device, load_config, load_json, save_json, set_seed


def make_weighted_sampler(csv_path: Path) -> WeightedRandomSampler:
    df = pd.read_csv(csv_path)
    class_counts = df["class_idx"].value_counts().to_dict()
    sample_weights = np.asarray(
        df["class_idx"].apply(lambda x: 1.0 / class_counts[int(x)]).values,
        dtype=np.float64,
    )
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights.copy()),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_loader(dataset, batch_size, shuffle=False, sampler=None, num_workers=2):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def make_class_weighted_losses(train_csv: Path, label_info: Dict, device: torch.device, tasks) -> Dict[str, nn.Module]:
    df = pd.read_csv(train_csv)
    losses = {}
    for task in tasks:
        idx_col = f"{task}_idx"
        num_labels = label_info[task]["num_classes"]
        counts = df[idx_col].value_counts().reindex(range(num_labels), fill_value=1).sort_index()
        weights = 1.0 / torch.tensor(counts.values, dtype=torch.float)
        weights = weights / weights.mean()
        losses[task] = nn.CrossEntropyLoss(weight=weights.to(device))
    return losses


def evaluate_one_epoch(model, dataloader, criterion_dict, loss_weights, device, tasks):
    model.eval()
    all_true = {task: [] for task in tasks}
    all_pred = {task: [] for task in tasks}
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = {task: batch["targets"][task].to(device) for task in tasks}
            outputs = model(images)
            loss = sum(loss_weights[task] * criterion_dict[task](outputs[task], targets[task]) for task in tasks)
            total_loss += loss.item() * images.size(0)
            for task in tasks:
                preds = outputs[task].argmax(dim=1).detach().cpu().numpy()
                truth = targets[task].detach().cpu().numpy()
                all_pred[task].extend(preds.tolist())
                all_true[task].extend(truth.tolist())
    metrics = {"loss": total_loss / len(dataloader.dataset)}
    for task in tasks:
        metrics[f"{task}_accuracy"] = accuracy_score(all_true[task], all_pred[task])
        metrics[f"{task}_macro_f1"] = f1_score(all_true[task], all_pred[task], average="macro", zero_division=0)
    metrics["joint_accuracy"] = np.mean([all(all_true[task][i] == all_pred[task][i] for task in tasks) for i in range(len(all_true["class"]))])
    return metrics


def train(args):
    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    processed_dir = Path(config["paths"]["processed_dir"])
    model_dir = ensure_dir(config["paths"]["model_dir"])
    metrics_dir = ensure_dir(config["paths"]["metrics_dir"])
    label_info = load_json(processed_dir / "labels.json")
    tasks = get_tasks(label_info)
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    device = get_device()
    print(f"Using device: {device}")

    train_ds = FashionCatalogDataset(train_csv, transform=build_transforms(config["training"]["image_size"], train=True))
    val_ds = FashionCatalogDataset(val_csv, transform=build_transforms(config["training"]["image_size"], train=False))

    sampler = make_weighted_sampler(train_csv) if config["training"].get("use_weighted_sampler", True) else None
    num_workers = config["training"]["num_workers"]
    train_loader = make_loader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        sampler=None,
        num_workers=num_workers,
    )

    model = build_model_from_label_info(label_info, args.backbone, config["training"]["dropout"], pretrained=not args.no_pretrained).to(device)
    print(f"Backbone: {args.backbone}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    criterion_dict = (
        make_class_weighted_losses(train_csv, label_info, device, tasks)
        if config["training"].get("use_class_weighted_loss", True)
        else {task: nn.CrossEntropyLoss() for task in tasks}
    )
    loss_weights = config["training"]["loss_weights"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

    best_val_macro_f1 = -1.0
    best_state = None
    patience = config["training"]["early_stopping_patience"]
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        start_time = time.perf_counter()
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}")
        for batch in loop:
            images = batch["image"].to(device)
            targets = {task: batch["targets"][task].to(device) for task in tasks}
            optimizer.zero_grad()
            outputs = model(images)
            loss = sum(loss_weights[task] * criterion_dict[task](outputs[task], targets[task]) for task in tasks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate_one_epoch(model, val_loader, criterion_dict, loss_weights, device, tasks)
        avg_val_macro_f1 = float(np.mean([val_metrics[f"{task}_macro_f1"] for task in tasks]))
        row = {"epoch": epoch, "train_loss": train_loss, "epoch_time_sec": round(time.perf_counter() - start_time, 3), "avg_val_macro_f1": avg_val_macro_f1, **val_metrics}
        history.append(row)
        print(row)

        if avg_val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = avg_val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    checkpoint = {"model_state_dict": best_state or model.state_dict(), "backbone": args.backbone, "label_info": label_info, "config": config, "image_size": config["training"]["image_size"]}
    torch.save(checkpoint, model_dir / "best_model.pt")
    torch.save(checkpoint, model_dir / f"best_model_{args.backbone}.pt")
    save_json({"history": history, "best_val_macro_f1": best_val_macro_f1}, metrics_dir / f"training_history_{args.backbone}.json")
    print(f"Saved checkpoint to: {model_dir / 'best_model.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50", "mobilenet_v3_large"])
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
