from __future__ import annotations
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _infer_repo_root(csv_path: Path) -> Path:
    """Repo root is parent of `data/` when splits live under `data/processed/`."""
    resolved = csv_path.resolve()
    if resolved.parent.name == "processed" and resolved.parent.parent.name == "data":
        return resolved.parent.parent.parent
    return Path.cwd()


def _resolve_image_path(image_path: Path, root_dir: Path) -> Path:
    if image_path.is_absolute():
        return image_path
    rooted = root_dir / image_path
    if rooted.exists():
        return rooted
    if image_path.exists():
        return image_path
    return rooted


class FashionCatalogDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None, root_dir: str | Path | None = None):
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir is not None else _infer_repo_root(self.csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        required_cols = ["image_path", "class_idx", "gender_idx", "color_idx", "usage_idx"]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Split file {csv_path} is missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        image_path = _resolve_image_path(Path(row["image_path"]), self.root_dir)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        targets = {
            "class": torch.tensor(int(row["class_idx"]), dtype=torch.long),
            "gender": torch.tensor(int(row["gender_idx"]), dtype=torch.long),
            "color": torch.tensor(int(row["color_idx"]), dtype=torch.long),
            "usage": torch.tensor(int(row["usage_idx"]), dtype=torch.long),
        }
        return {"image": image, "targets": targets, "image_path": str(image_path)}
