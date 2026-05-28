# Fashion Catalog Enrichment Pipeline

End-to-end computer vision prototype for retail catalog enrichment. Upload product images and get structured predictions for **Class**, **Gender**, **Color**, and **Usage** (bonus attribute) from a single multi-head model.

**Status:** Code on GitHub, Streamlit demo deployable on EC2. Three ImageNet-pretrained backbones trained and compared. **MobileNetV3 Large** is the quality leader on the latest 4-head benchmark; **ONNX Runtime** on EC2 CPU gives up to **2.42×** lower latency than PyTorch with identical accuracy.

## What the system predicts

| Head | Labels (examples) |
|------|-------------------|
| Class | Shirt, T-Shirt, Pants, Shoes, Watch, Sunglasses, Bag, Cap |
| Gender | Men, Women, Kids, Unisex |
| Color | Black, Blue, White, Red, Green, Grey, Brown, Pink, Yellow, Other (+ rare colors grouped) |
| Usage | Casual, Sports, Formal, Ethnic, Other |

Each head returns a label, confidence score, and per-image **TAT** (turnaround time). **Joint accuracy** requires all four heads to be correct on the same image.

## Architecture

V1 uses **transfer learning** with **fine-tuning** (pretrained backbone + four heads train together from the start—not staged freeze/unfreeze).

```text
Input image (224×224)
   ↓
Pretrained CNN backbone (ImageNet weights)
   ↓
Shared embedding
   ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Class   │ Gender  │ Color   │ Usage   │
│ head    │ head    │ head    │ head    │
└─────────┴─────────┴─────────┴─────────┘
   ↓
Structured JSON + confidences + TAT
```

**Backbones compared:** EfficientNet-B0, ResNet50, MobileNetV3 Large.

**Loss (weighted multi-task CE):**

```text
Total = 1.0×Class + 0.7×Gender + 1.2×Color + 0.6×Usage
```

**Imbalance handling:** category filtering, optional per-class caps, `WeightedRandomSampler` on class, class-weighted CE, macro F1 for evaluation.

**Train augmentations:** horizontal flip, rotation (±10°), color jitter, resize 224, ImageNet normalize.

Future extension for cluttered images (not in deployed V1):

```text
Raw image → YOLOv8 / Detectron2 crop → multi-head classifier
```

## Model selection (test set, 4-head)

Quality is **identical on Mac and EC2** (same checkpoints and test split). Latency differs by hardware and engine—see [Deployment](#deployment-mac-vs-ec2).

| Backbone | Class | Gender | Color | Usage | Joint | Color F1 | Val macro F1 |
|----------|-------|--------|-------|-------|-------|----------|--------------|
| EfficientNet-B0 | 98.63% | 88.24% | 63.22% | 85.68% | 47.36% | 57.46% | 74.13% |
| ResNet50 | 98.63% | 86.42% | 62.48% | 86.26% | 47.17% | 57.75% | 74.39% |
| **MobileNetV3 Large** | 97.99% | 86.71% | **65.04%** | **86.67%** | **48.29%** | **59.92%** | **74.79%** |

**Recommendation:** Use **MobileNetV3 Large** when color and joint accuracy matter most. **ResNet50** remains a useful comparison / fallback. Class accuracy is already >97% for all models—optimize for **color** and **joint** metrics.

> Joint accuracy is lower than the older 3-head model because all **four** heads must match; do not compare 3-head and 4-head joint numbers directly.

### EC2 latency (mean ms/image, batch=32)

| Model | EC2 PyTorch | EC2 ONNX | Speedup |
|-------|-------------|----------|---------|
| EfficientNet-B0 | 22.2 | 18.8 | 1.18× |
| ResNet50 | 68.9 | 35.3 | 1.96× |
| **MobileNetV3 Large** | 16.5 | **6.8** | **2.42×** |

On **Mac**, PyTorch with **MPS** is fastest; ONNX on Mac uses CPU and is slower than MPS PyTorch.

## Repository structure

```text
fashion_catalog_enrichment_1/
├── app/
│   └── streamlit_app.py          # Demo: upload/samples, PyTorch or ONNX, review queue
├── configs/
│   └── config.yaml               # Data, training, loss weights, paths
├── data/
│   ├── raw/                      # styles.csv + images/ (not in git; download separately)
│   └── processed/                # train/val/test CSVs, labels.json, EDA summary
├── notebooks/
│   ├── 02_training_walkthrough.ipynb
│   └── mac_vs_ec2_perf.ipynb
├── outputs/
│   ├── models/
│   │   ├── best_model_<backbone>.pt
│   │   └── onnx/<backbone>/model.onnx
│   ├── metrics/                  # evaluation JSON, comparison CSV, benchmarks
│   └── review_queue/
├── scripts/
│   └── download_kaggle_dataset.sh
├── src/
│   ├── data_preparation.py       # Filter, map labels, stratified split
│   ├── label_mapping.py
│   ├── dataset.py
│   ├── model.py                  # Multi-head CNN + backbones
│   ├── tasks.py
│   ├── train.py / train_all.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── inference.py              # PyTorch batch inference
│   ├── onnx_inference.py
│   ├── optimize_onnx.py          # Export + optional INT8 quantize
│   ├── benchmark_latency.py
│   ├── benchmark_engines.py
│   ├── benchmark_all_models_engines.py
│   ├── active_learning.py
│   └── utils.py                  # Device (cuda / mps / cpu)
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── REPORT_TEMPLATE.md
└── README.md
```

**Not committed to git:** `data/raw/images/`, trained `.pt` checkpoints, large metric artifacts. Copy to servers via `rsync`/`scp` (see `notebooks/DEPLOYMENT_PLAN.md`).

## Dataset

**Fashion Product Images Small** (Kaggle): product images plus metadata (`articleType`, `gender`, `baseColour`, `usage`, etc.).

**Preparation pipeline:**

1. Load metadata and filter to 8 retail classes (min samples per class).
2. Verify images; drop corrupt files.
3. Normalize gender, color (rare → families), usage.
4. Stratified **70 / 15 / 15** train/val/test split by class.
5. Encode labels for all four heads → `data/processed/labels.json`.

Expected raw layout:

```text
data/raw/
├── styles.csv
└── images/
    ├── 1163.jpg
    └── ...
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download Kaggle data

```bash
pip install kaggle
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/ and chmod 600
bash scripts/download_kaggle_dataset.sh
```

## Workflow

### 1. Prepare data

```bash
python -m src.data_preparation --config configs/config.yaml
```

Outputs: `data/processed/{train,val,test}.csv`, `labels.json`, `eda_summary.json`.

### 2. Train

All three backbones:

```bash
python -m src.train_all --config configs/config.yaml
```

Single backbone:

```bash
python -m src.train --config configs/config.yaml --backbone mobilenet_v3_large
```

Skip existing checkpoints:

```bash
python -m src.train_all --skip-existing
```

Checkpoints: `outputs/models/best_model_<backbone>.pt`  
History: `outputs/metrics/training_history_<backbone>.json`

Training defaults (`configs/config.yaml`): 5 epochs, early stopping patience 3, AdamW lr `3e-4`, batch 32, weighted sampler on class.

### 3. Evaluate and compare

Per model:

```bash
python -m src.evaluate \
  --checkpoint outputs/models/best_model_mobilenet_v3_large.pt \
  --split test \
  --tag mobilenet_v3_large
```

All models (table + `model_comparison_test.csv`):

```bash
python -m src.compare_models --config configs/config.yaml --split test
```

Metrics: per-head accuracy and macro F1, joint accuracy, confusion matrices, validation macro F1.

### 4. Export ONNX (EC2 / CPU serving)

Batched float32 ONNX (recommended for Streamlit batch inference on EC2):

```bash
python -m src.optimize_onnx \
  --checkpoint outputs/models/best_model_mobilenet_v3_large.pt \
  --output_dir outputs/models/onnx/mobilenet_v3_large \
  --dynamic-batch \
  --no-quantize
```

Optional dynamic INT8 (smaller; often static batch=1):

```bash
python -m src.optimize_onnx \
  --checkpoint outputs/models/best_model_mobilenet_v3_large.pt \
  --output_dir outputs/models/onnx/mobilenet_v3_large
```

### 5. Benchmark latency (PyTorch vs ONNX)

All backbones and engines on test split:

```bash
python -m src.benchmark_all_models_engines \
  --split test \
  --batch-size 32 \
  --max-batches 0 \
  --onnx-root outputs/models/onnx
```

Writes `outputs/metrics/engine_benchmark_summary_test.csv` (and per-model JSON).

### 6. Streamlit demo

```bash
streamlit run app/streamlit_app.py --server.port 8789 --server.address 0.0.0.0
```

Features:

- Backbone dropdown (EfficientNet-B0, ResNet50, MobileNetV3 Large)
- Engine: **PyTorch** (Mac MPS / CUDA) or **ONNX Runtime** (batched float32 at `outputs/models/onnx/<backbone>/model.onnx`)
- True **batch inference** for multiple uploads
- Per-attribute confidence, TAT, top-3, JSON export
- Low-confidence **review queue** CSV

### Sample JSON output

```json
{
  "image_name": "shirt_1.jpg",
  "prediction": {
    "class": {"label": "Shirt", "confidence": 0.94},
    "gender": {"label": "Men", "confidence": 0.89},
    "color": {"label": "Blue", "confidence": 0.82},
    "usage": {"label": "Casual", "confidence": 0.91}
  },
  "tat_ms": 172.4
}
```

## Deployment: Mac vs EC2

| Environment | Recommended engine | Why |
|-------------|-------------------|-----|
| Mac (Apple Silicon) | PyTorch + **MPS** | Fastest local inference |
| EC2 (CPU) | **ONNX Runtime** | 1.18×–2.42× faster than PyTorch CPU; same accuracy |

Deploy steps and `rsync` commands: `notebooks/DEPLOYMENT_PLAN.md`.

## Active learning

Flag low-confidence predictions for human review:

```bash
python -m src.active_learning \
  --predictions_json outputs/metrics/sample_predictions.json \
  --output_csv outputs/review_queue/review_queue.csv \
  --threshold 0.70
```

## Evaluation beyond accuracy

This project reports:

- Per-head **macro F1** (minority labels, especially color and usage)
- **Joint accuracy** (all four heads correct)
- Multi-backbone comparison
- Mac vs EC2 **quality parity** and **latency** benchmarks
- Planned: color confusion analysis, P95 TAT, external validation sample

V1 does **not** report IoU/mAP—those apply to a detection stage (YOLOv8/Detectron2 POC in the 2-week plan).

## Production scaling (outline)

```text
S3/GCS images → metadata DB → queue (SQS/Kafka) → GPU/ONNX batch workers
  → enrichment table → low-confidence review → retrain with corrections
```

Design choices: batch inference, horizontal workers, model versioning, per-attribute confidence thresholds, drift monitoring, FastAPI or Triton + ONNX on CPU/GPU.

## Roadmap (2-week plan)

- Higher-quality, less pixelated training data + manual color validation set
- Color loss-weight sweep and confusion-pair analysis
- Dominant-color fallback for low-confidence color
- Optional color-only specialist + ensemble
- YOLOv8 / Detectron2 localization POC
- INT8 ONNX and pruning if size/latency still block
- Active learning script integration with review queue

Details: `notebooks/PLANNED_FEATURES.md`, `notebooks/EXPERIMENT_TIMELINE.md`.

## Further reading

| Document | Contents |
|----------|----------|
| `REPORT_TEMPLATE.md` | Assignment-style report skeleton |
| `notebooks/EXPERIMENT_TIMELINE.md` | Experiment log and metrics history |
| `notebooks/DEPLOYMENT_PLAN.md` | EC2 setup, copy commands, Streamlit |
| `notebooks/ML_BASICS_REVISION.md` | Tensors, CNNs, training concepts |

## License

See `LICENSE_NOTE.md`.
