# Report: Fashion Catalog Enrichment Pipeline

## 1. Problem Statement

The objective is to build an end-to-end computer vision pipeline that accepts raw retail product images and returns structured attributes for catalog enrichment.

Mandatory outputs:

- Class
- Gender
- Color

Additional outputs:

- Confidence scores
- Turnaround Time per image
- Optional retail attributes in future versions

## 2. Dataset

Dataset used: Fashion Product Images Small.

Metadata columns used:

- id
- gender
- articleType
- baseColour
- masterCategory
- subCategory
- usage
- season
- productDisplayName

## 3. Data Preparation

Steps performed:

1. Loaded metadata CSV.
2. Mapped article types into business-friendly classes.
3. Filtered selected product categories.
4. Verified image files exist.
5. Removed unreadable/corrupted images.
6. Normalized gender and color labels.
7. Grouped rare colors into `Other`.
8. Created stratified train/validation/test split.

## 4. Sampling Strategy and Class Imbalance

Class imbalance was handled through:

- Category selection with minimum sample thresholds.
- Optional cap on over-represented classes.
- WeightedRandomSampler during training.
- Class-weighted loss.
- Macro F1 as a core metric.

This prevents large classes from dominating the final score.

## 5. Model Architecture

The model uses transfer learning with a shared visual backbone and three classification heads.

```text
Image
  ↓
EfficientNet-B0 / ResNet50 backbone
  ↓
Shared embedding
  ↓
Class Head
Gender Head
Color Head
```

## 6. Localization vs Classification

The assignment mentions localize-then-classify as an option.

For this version, a direct classifier was selected because the dataset contains mostly single-centered catalog images.

A future real-world version can add:

```text
YOLOv8 detector → crop detected product → multi-head classifier
```

This will help for cluttered images, user-generated images, or images with multiple products.

## 7. Evaluation Strategy

The evaluation goes beyond accuracy.

| Task | Metrics |
|---|---|
| Class | Accuracy, Macro F1, Weighted F1, Per-class F1 |
| Gender | Accuracy, Macro F1, Weighted F1 |
| Color | Accuracy, Macro F1, Weighted F1, confusion matrix |
| Full prediction | Joint exact match for class + gender + color |
| Latency | Mean TAT, median TAT, p95 TAT |

## 8. Model Comparison

| Model | Class Macro F1 | Gender Macro F1 | Color Macro F1 | Avg TAT ms | Model Size MB |
|---|---:|---:|---:|---:|---:|
| ResNet50 | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 ONNX | TBD | TBD | TBD | TBD | TBD |
| EfficientNet-B0 INT8 ONNX | TBD | TBD | TBD | TBD | TBD |

Decision:

> The final model should be selected based on a practical balance between macro F1 and inference latency.

## 9. Optimization

Optimization techniques used/planned:

- Batch inference for up to 10 images.
- `torch.no_grad()` during inference.
- ONNX export.
- ONNX Runtime inference.
- Optional dynamic INT8 quantization.
- Model loading cache in Streamlit.

## 10. Working Prototype

The web app supports:

- Upload up to 10 images.
- View predicted labels.
- View confidence scores.
- View per-image TAT.
- Download JSON predictions.
- Export low-confidence predictions to active learning queue.

## 11. Production Philosophy: Scaling to 1 Million Images

A production-grade system would be designed as follows:

```text
Object storage: S3/GCS
   ↓
Metadata table
   ↓
Queue: SQS/Kafka/PubSub
   ↓
GPU inference workers
   ↓
Prediction database
   ↓
Catalog enrichment service
   ↓
Review queue for low-confidence predictions
```

Key production practices:

- Batch inference to maximize GPU utilization.
- Horizontal scaling of inference workers.
- Confidence thresholds by attribute.
- Monitoring of class drift and label drift.
- Human review for ambiguous predictions.
- Continuous retraining from reviewed examples.
- Separate offline batch pipeline and online real-time API.

## 12. Learnings and Trade-offs

Key trade-offs to document after experiments:

1. ResNet50 may offer stronger accuracy but higher latency.
2. EfficientNet-B0 may offer better TAT with acceptable accuracy.
3. Direct classification works for catalog-style images.
4. Detection is useful but not required for clean single-product images.
5. Color prediction can be noisy due to lighting, background, and multi-color products.
6. Joint exact match is stricter than individual task accuracy.

## 13. Active Learning Next Steps

If given two more weeks:

1. Add review UI for low-confidence predictions.
2. Add human correction storage.
3. Retrain model using corrected labels.
4. Improve rare color and rare class handling.
5. Add YOLOv8 localization for cluttered images.
6. Add fine-grained attributes such as sleeve length, neckline, material, and pattern.
