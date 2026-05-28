from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.label_mapping import make_label_maps, map_article_type, map_color, map_gender, map_usage
from src.utils import ensure_dir, load_config, save_json, set_seed

REQUIRED_COLUMNS = ["id", "gender", "articleType", "baseColour", "usage"]


def _image_path_for_id(image_dir: Path, image_id: object) -> Path:
    return image_dir / f"{int(image_id)}.jpg"


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")


def prepare_metadata(config: Dict) -> Tuple[pd.DataFrame, Dict]:
    paths = config["paths"]
    data_cfg = config["data"]
    raw_csv = Path(paths["raw_metadata_csv"])
    image_dir = Path(paths["raw_image_dir"])

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"Could not find {raw_csv}.\n"
            "Download the dataset first:\n"
            "  bash scripts/download_kaggle_dataset.sh\n"
            "Or: kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/raw --unzip\n"
            "Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small"
        )
    if not image_dir.exists():
        raise FileNotFoundError(f"Could not find {image_dir}. Place images under data/raw/images or update config.")

    df = pd.read_csv(raw_csv, on_bad_lines="skip")
    _validate_columns(df)
    df = df.copy()
    df["image_path"] = df["id"].apply(lambda x: str(_image_path_for_id(image_dir, x)))
    df["class_label"] = df["articleType"].apply(lambda value: map_article_type(value, data_cfg["selected_article_types"]))
    df["gender_label"] = df["gender"].apply(lambda value: map_gender(value, data_cfg["gender_mapping"]))
    df["color_label"] = df["baseColour"].apply(lambda value: map_color(value, data_cfg["color_keep_list"], data_cfg["rare_color_label"]))
    df["usage_label"] = df["usage"].apply(
        lambda value: map_usage(value, data_cfg["usage_mapping"], data_cfg.get("rare_usage_label", "Other"))
    )
    df = df.dropna(subset=["class_label", "gender_label", "color_label"])
    df = df[df["image_path"].apply(lambda path: Path(path).exists())]

    if data_cfg.get("verify_images", True):
        tqdm.pandas(desc="Verifying images")
        df = df[df["image_path"].progress_apply(lambda path: _is_readable_image(Path(path)))]

    class_counts = df["class_label"].value_counts()
    keep_classes = class_counts[class_counts >= data_cfg["min_samples_per_class"]].index.tolist()
    df = df[df["class_label"].isin(keep_classes)]

    max_samples = data_cfg.get("max_samples_per_class")
    if max_samples:
        sampled_groups = []
        for _, group in df.groupby("class_label", sort=False):
            sampled_groups.append(
                group.sample(n=min(len(group), max_samples), random_state=config["project"]["seed"])
            )
        df = pd.concat(sampled_groups, ignore_index=True)

    df = df.reset_index(drop=True)
    label_info = {
        "class": make_label_maps(df["class_label"].tolist()),
        "gender": make_label_maps(df["gender_label"].tolist()),
        "color": make_label_maps(df["color_label"].tolist()),
        "usage": make_label_maps(df["usage_label"].tolist()),
    }
    for task in ["class", "gender", "color", "usage"]:
        df[f"{task}_idx"] = df[f"{task}_label"].map(label_info[task]["label_to_idx"])

    eda_summary = {
        "total_rows": int(len(df)),
        "class_distribution": df["class_label"].value_counts().to_dict(),
        "gender_distribution": df["gender_label"].value_counts().to_dict(),
        "color_distribution": df["color_label"].value_counts().to_dict(),
        "usage_distribution": df["usage_label"].value_counts().to_dict(),
        "num_classes": label_info["class"]["num_classes"],
        "num_genders": label_info["gender"]["num_classes"],
        "num_colors": label_info["color"]["num_classes"],
        "num_usage_labels": label_info["usage"]["num_classes"],
    }
    return df, {"label_info": label_info, "eda_summary": eda_summary}


def create_splits(df: pd.DataFrame, config: Dict) -> Dict[str, pd.DataFrame]:
    data_cfg = config["data"]
    seed = config["project"]["seed"]
    test_size = float(data_cfg["test_size"])
    val_size = float(data_cfg["val_size"])
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["class_label"])
    val_fraction_of_train_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_fraction_of_train_val, random_state=seed, stratify=train_val_df["class_label"])
    return {"train": train_df.reset_index(drop=True), "val": val_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    processed_dir = ensure_dir(config["paths"]["processed_dir"])
    df, artifacts = prepare_metadata(config)
    splits = create_splits(df, config)
    df.to_csv(processed_dir / "metadata_processed.csv", index=False)
    for split_name, split_df in splits.items():
        split_df.to_csv(processed_dir / f"{split_name}.csv", index=False)
    save_json(artifacts["label_info"], processed_dir / "labels.json")
    save_json(artifacts["eda_summary"], processed_dir / "eda_summary.json")
    print("Data preparation completed.")
    print(json.dumps(artifacts["eda_summary"], indent=2))


if __name__ == "__main__":
    main()
