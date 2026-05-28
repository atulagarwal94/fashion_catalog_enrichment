# Experiment timeline — changes, metrics, and why

> A “what changed / why / what moved” record based on our chat + saved artifacts in `outputs/metrics/`.  
> This doc is intentionally **evidence-based**: every metric number comes from a file in the repo.  
> Last updated: 2026-05-28

---

## TL;DR

- **Core story**: started with a 3-head model (**class/gender/color**), then improved training speed (MPS), demo UX (Streamlit), inference latency (true batch), and finally added a **4th head (`usage`)**.
- **Main bottleneck** across all experiments: **color** (sensitive to lighting / similar labels like Blue vs Navy Blue).
- **Joint accuracy** definition changed:
  - **Before**: exact match across **3 heads** (class+gender+color)
  - **Now**: exact match across **4 heads** (class+gender+color+usage) → will be lower by design.

---

## How to read metrics in this doc

- **Test-set comparison (latest)** lives in:
  - `outputs/metrics/model_comparison_test.csv`
  - `outputs/metrics/model_comparison_test.json`
- **Per-model evaluation (latest)** lives in:
  - `outputs/metrics/evaluation_test_<backbone>.json`
  - `outputs/metrics/classification_report_<task>_test_<backbone>.csv`
- **Older baselines can get overwritten** if we re-run training/eval without saving tags. Where that happened, this doc explicitly says so.

---

## 2026-05-26 — Fix notebook environment + file paths

### Change
- Rebuilt `.venv` after project path move (“bad interpreter”).
- Fixed `FileNotFoundError` for images when notebook cwd was `notebooks/`.

### Why
- Notebook kernel and dataset loading were broken; CSV image paths were relative to repo root.

### Key files
- `notebooks/README.md`
- `src/dataset.py` (path resolution via repo root inference)
- `notebooks/02_training_walkthrough.ipynb` (chdir to repo root)

### Metric impact
- **N/A** (reliability fix; no model change).

---

## 2026-05-26 — Training speed improvements (Apple Silicon)

### Change
- Device selection updated to support **MPS** (Apple GPU) in `get_device()`.
- DataLoader tuning: `persistent_workers`, `prefetch_factor`, `num_workers` increased.

### Why
- Training was measured at ~49 min/epoch on CPU; MPS was available but unused.

### Key files
- `src/utils.py`
- `src/train.py`
- `configs/config.yaml` (`training.num_workers`)

### Metric impact
- Primary impact is **runtime** (epoch time), not accuracy.
- **Note:** historical latency benchmark artifacts were overwritten (no `latency_benchmark.json` currently in repo).

---

## 2026-05-26 — Multi-backbone training + test comparison (3-head baseline)

### Change
- Added terminal workflow to train all 3 backbones and compare them on test.

### Why
- Needed consistent comparison of backbones (EfficientNet vs ResNet vs MobileNet).

### Key files
- `src/train_all.py`
- `src/evaluate.py` (added `--tag` to avoid overwriting)
- `src/compare_models.py`
- `README.md` (commands)

### Metrics (baseline, **3 heads**, pre-usage)

From `outputs/metrics/evaluation_test.json` (3-head exact match joint):

- Class accuracy: **98.72%**
- Gender accuracy: **89.55%**
- Color accuracy: **65.64%**
- Joint accuracy (class+gender+color): **58.52%**

Evidence:
- `outputs/metrics/evaluation_test.json`

### Takeaway
- **Class** was already near-solved (~99%).
- **Color** was the weak head (~66%) and capped joint accuracy.

---

## 2026-05-26 → 2026-05-27 — Streamlit demo upgrades

### Change
- Added richer UI: confidence badges, top-3, review queue download, sidebar metrics, etc.
- Added model dropdown (instead of manual checkpoint path).

### Why
- Improve demo usability; make model selection and “low confidence → review” workflow clear.

### Key files
- `app/streamlit_app.py`
- `src/inference.py`
- `src/active_learning.py`

### Metric impact
- **Accuracy unchanged** (same checkpoints); improves demo/ops only.

---

## 2026-05-27 — True batch inference (latency / TAT improvement)

### Change
- Implemented `predict_batch_pil()`:
  - stack \(N\) images → `[N,3,224,224]`
  - **one** forward pass
  - decode \(N\) outputs
- Streamlit switched to **one forward pass per submit** (amortized per-image TAT).

### Why
- Demo previously did one forward per image; batch improves average latency and matches training/eval batching.

### Key files
- `src/inference.py` (`predict_batch_pil`)
- `app/streamlit_app.py` (batch call)
- `src/benchmark_latency.py` (updated to use true batch)

### Metric impact
- **Accuracy unchanged** (same weights).
- Latency improves, but **no saved `latency_benchmark.json` exists right now**; rerun after training to record:

```bash
python -m src.benchmark_latency --checkpoint outputs/models/best_model_resnet50.pt
```

---

## 2026-05-27 — Loss-weight tuning: color emphasized

### Change
- Updated `configs/config.yaml`:
  - `training.loss_weights.color`: **0.8 → 1.2**

### Why
- Color was weakest; increasing weight pushes optimizer to spend more capacity on color head.

### Key file
- `configs/config.yaml`

### Metric impact (important note)
- This change only affects models **after retraining**.
- Our earlier “3-head baseline” metrics are preserved in `outputs/metrics/evaluation_test.json`, but after adding usage the pipeline became 4-head and the newer evaluations are not directly comparable on joint accuracy.

---

## 2026-05-27 — Added `usage` attribute (4th head)

### Change
- Added `usage` label from `styles.csv` and trained as a **4th head**.
- Joint accuracy now means **class+gender+color+usage** exact match.

### Why
- Bonus attribute; `usage` exists in raw metadata and strengthens “catalog enrichment” story.

### Key files
- Data: `src/data_preparation.py`, `src/label_mapping.py`, `configs/config.yaml`
- Model: `src/model.py` (+ optional `usage_head`)
- Dataset: `src/dataset.py` (adds `usage_idx` target)
- Train/eval: `src/train.py`, `src/evaluate.py`, `src/compare_models.py`
- Inference/UI: `src/inference.py`, `app/streamlit_app.py`

### Metrics (latest, **4 heads**, post-usage)

From `outputs/metrics/model_comparison_test.csv`:

| Backbone | Class | Gender | Color | Usage | Joint (4-way) |
|----------|------:|-------:|------:|------:|--------------:|
| efficientnet_b0 | 98.63% | 88.24% | 63.22% | 85.68% | 47.36% |
| resnet50 | 98.63% | 86.42% | 62.48% | 86.26% | 47.17% |
| mobilenet_v3_large | 97.99% | 86.71% | 65.04% | 86.67% | 48.29% |

Evidence:
- `outputs/metrics/model_comparison_test.csv`
- Per-model JSONs: `outputs/metrics/evaluation_test_<backbone>.json`

### What improved / what didn’t
- **Usage accuracy** is solid (~85–87%), but **usage macro F1 is ~0.48** because the label distribution is extremely imbalanced.
  - Example (ResNet50 usage): `outputs/metrics/classification_report_usage_test_resnet50.csv`
  - `Ethnic` and `Other` supports are tiny → macro-F1 penalized.
- **Color accuracy** is still the hardest head (~62–65%).
- **Joint (4-way)** dropped vs 3-way baseline because it’s a stricter metric (one more condition to satisfy).

### Notes on label imbalance (usage)
- From data prep summary (printed during `python -m src.data_preparation`):
  - Casual dominates; Ethnic/Other have tiny supports.
- In test report for ResNet50, `Ethnic` support is **3**, `Other` support is **2**:
  - `outputs/metrics/classification_report_usage_test_resnet50.csv`

---

## 2026-05-28 — Streamlit UX: “Run prediction” button + default ResNet50 + built-in sample images

### Change
- Added **Run prediction** button (no auto prediction when selecting images).
- Default backbone in dropdown is now **ResNet50** (if present).
- Added “Built-in sample images” mode; reads from `app/sample_images/` (or fallback under `app/`).

### Why
- User-controlled inference; smoother demo on EC2; no need to hunt for images locally.

### Key file
- `app/streamlit_app.py`

### Metric impact
- Accuracy unchanged; improves demo workflow.

---

## What changed metrics the most (so far)

1. **Adding `usage` head**: joint accuracy became stricter (4-way), so joint % dropped vs 3-way baseline by definition.
2. **Loss weight changes** (color 0.8 → 1.2): intended to move color metrics, but clean “before vs after” requires keeping separate evaluation artifacts per run (see next section).

---

## Recommendation: avoid overwriting metrics going forward

If we want clean “experiment A vs experiment B” comparisons (e.g., color weight 0.8 vs 1.2), we should save outputs with a run tag, e.g.:

- `evaluation_test_resnet50_color08.json`
- `evaluation_test_resnet50_color12.json`

Today we already have `--tag` in `src.evaluate`; we can extend this to include config tags in filenames.

---

## Appendix — Evidence index (files used)

- **3-head baseline**
  - `outputs/metrics/evaluation_test.json`
- **4-head latest**
  - `outputs/metrics/model_comparison_test.csv`
  - `outputs/metrics/model_comparison_test.json`
  - `outputs/metrics/evaluation_test_resnet50.json`
  - `outputs/metrics/classification_report_usage_test_resnet50.csv`
- **Change log**
  - `notebooks/PROJECT_CHANGELOG.md`
- **Deployment**
  - `notebooks/DEPLOYMENT_PLAN.md`

