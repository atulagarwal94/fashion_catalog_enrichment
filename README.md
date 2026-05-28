# Fashion Catalog Enrichment: Multi-Task Computer Vision Pipeline

This repository contains an end-to-end computer vision prototype for retail catalog enrichment.

The system accepts fashion/product images and returns structured predictions for:

- **Class**: Shirt, T-Shirt, Pants, Shoes, Watch, Sunglasses, Bag, Cap
- **Gender**: Men, Women, Kids, Unisex
- **Color**: Black, Blue, White, Red, etc.
- **Confidence scores**
- **Turnaround Time (TAT)** per image

## Architecture

```text
Input image
   ↓
Shared CNN backbone: EfficientNet-B0 or ResNet50
   ↓
Class Head  → Product class
Gender Head → Gender
Color Head  → Base color
   ↓
Structured JSON output
```

For V1, the project uses direct image classification because catalog datasets usually contain one centered product. For production images with background clutter, the future extension is:

```text
Raw image → YOLOv8 product detector → Product crop → Multi-head classifier
```

## Repository Structure

```text
fashion_catalog_enrichment/
├── app/streamlit_app.py
├── configs/config.yaml
├── data/raw/
├── data/processed/
├── outputs/models/
├── outputs/metrics/
├── outputs/review_queue/
├── scripts/download_kaggle_dataset.sh
├── src/
│   ├── active_learning.py
│   ├── benchmark_latency.py
│   ├── data_preparation.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── label_mapping.py
│   ├── model.py
│   ├── optimize_onnx.py
│   ├── train.py
│   └── utils.py
├── tests/test_smoke.py
├── requirements.txt
└── REPORT_TEMPLATE.md
```

## Dataset

Recommended dataset: **Fashion Product Images Small** from Kaggle.

Expected raw structure:

```text
data/raw/
├── styles.csv
└── images/
    ├── 1163.jpg
    ├── 1164.jpg
    └── ...
```

Expected metadata columns:

```text
id, gender, masterCategory, subCategory, articleType, baseColour, season, usage, productDisplayName
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Download Kaggle Dataset

```bash
pip install kaggle
mkdir -p ~/.kaggle
# place kaggle.json inside ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
bash scripts/download_kaggle_dataset.sh
```

## 1. Prepare Data

```bash
python -m src.data_preparation --config configs/config.yaml
```

Outputs:

```text
data/processed/metadata_processed.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/labels.json
data/processed/eda_summary.json
```

## 2. Train Models

Train all three backbones in one command:

```bash
python -m src.train_all --config configs/config.yaml
```

Skip backbones that are already trained:

```bash
python -m src.train_all --skip-existing
```

Train only specific backbones:

```bash
python -m src.train_all --backbones resnet50 mobilenet_v3_large
```

Train EfficientNet-B0 only:

```bash
python -m src.train --config configs/config.yaml --backbone efficientnet_b0
```

Train ResNet50 for comparison:

```bash
python -m src.train --config configs/config.yaml --backbone resnet50
```

Outputs:

```text
outputs/models/best_model.pt
outputs/models/best_model_efficientnet_b0.pt
outputs/models/best_model_resnet50.pt
outputs/metrics/training_history_<backbone>.json
```

## 3. Evaluate

Single model:

```bash
python -m src.evaluate \
  --config configs/config.yaml \
  --checkpoint outputs/models/best_model.pt \
  --split test
```

Per-backbone outputs (keeps separate confusion matrices and JSON):

```bash
python -m src.evaluate \
  --checkpoint outputs/models/best_model_resnet50.pt \
  --split test \
  --tag resnet50
```

Compare all three models on the same test split (table + CSV):

```bash
python -m src.compare_models --config configs/config.yaml --split test
```

Re-print comparison from saved metrics only:

```bash
python -m src.compare_models --skip-eval
```

Writes `outputs/metrics/model_comparison_test.csv` and per-model `evaluation_test_<backbone>.json`.

Tracked metrics:

- Accuracy per task
- Macro F1 per task
- Weighted F1 per task
- Per-class F1
- Joint exact match across class + gender + color
- Confusion matrix

## 4. Benchmark Latency

```bash
python -m src.benchmark_latency \
  --checkpoint outputs/models/best_model.pt \
  --image_dir data/sample_images \
  --batch_size 10
```

Output:

```text
outputs/metrics/latency_benchmark.json
```

## 5. Export to ONNX and Quantize

```bash
python -m src.optimize_onnx \
  --checkpoint outputs/models/best_model.pt \
  --output_dir outputs/models
```

Outputs:

```text
outputs/models/model.onnx
outputs/models/model_quantized.onnx
outputs/metrics/onnx_optimization_report.json
```

## 6. Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app supports:

- Uploading up to 10 images
- Prediction labels
- Confidence scores
- TAT per image
- JSON output
- JSON download
- Low-confidence review queue export

## Sample JSON Output

```json
{
  "image_name": "shirt_1.jpg",
  "prediction": {
    "class": {"label": "Shirt", "confidence": 0.9412},
    "gender": {"label": "Men", "confidence": 0.8874},
    "color": {"label": "Blue", "confidence": 0.8169}
  },
  "tat_ms": 172.4
}
```

## Active Learning

The repo includes a confidence-based active learning queue.

```bash
python -m src.active_learning \
  --predictions_json outputs/metrics/sample_predictions.json \
  --output_csv outputs/review_queue/review_queue.csv \
  --threshold 0.70
```

Logic:

```text
If class/gender/color confidence < threshold:
    save image name + predicted labels + confidence to review_queue.csv
```

## Production Scaling Plan

For 1 million images:

```text
Images in S3/GCS
   ↓
Image metadata table
   ↓
Message queue: SQS/Kafka/PubSub
   ↓
GPU batch inference workers
   ↓
Prediction table
   ↓
Low-confidence review queue
   ↓
Human correction
   ↓
Retraining dataset
```

Production considerations:

- GPU batch inference workers
- ONNX Runtime or NVIDIA Triton for optimized serving
- Separate real-time inference and offline batch enrichment
- Drift monitoring by class, gender, and color
- Human review for low-confidence predictions
- Periodic retraining with reviewed labels

## Recommended Trade-off Experiments

| Experiment | Why |
|---|---|
| ResNet50 multi-head | Strong baseline accuracy |
| EfficientNet-B0 multi-head | Faster lightweight model |
| EfficientNet-B0 + ONNX | TAT optimization |
| EfficientNet-B0 + INT8 ONNX | Model-size and latency trade-off |

Final model should be selected based on macro F1 + latency, not accuracy alone.
