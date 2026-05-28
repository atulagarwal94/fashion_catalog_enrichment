# Project change log (timeline)

> Living record of **what changed**, **why**, and **where** in this repo.  
> Update this file when you merge meaningful work (training runs, config changes, new scripts).

---

## How to use

| Column | Meaning |
|--------|---------|
| **What** | Concrete change |
| **Why** | Problem it solved or goal |
| **Files** | Main paths touched |
| **Notes** | Commands, caveats, follow-ups |

---

## 2026-05-26 — Environment & notebook setup

### Jupyter / venv fixes

| | |
|---|---|
| **What** | Documented notebook setup; fixed broken `.venv` (`bad interpreter` after project path move). |
| **Why** | Notebook kernel could not run; paths pointed at old `fashion_catalog_enrichment` directory. |
| **Files** | `notebooks/README.md`, recreated `.venv` |
| **Notes** | Use repo-root cwd in notebooks; register ipykernel from project venv. |

### Image path resolution (`FileNotFoundError`)

| | |
|---|---|
| **What** | CSV image paths resolved relative to repo root; `FashionCatalogDataset` accepts `root_dir`. |
| **Why** | Notebook cwd was `notebooks/` so relative paths like `data/raw/images/...` failed. |
| **Files** | `src/dataset.py`, `notebooks/02_training_walkthrough.ipynb` (setup / chdir) |
| **Notes** | `os.chdir(REPO_ROOT)` in notebook setup cells. |

### Learning doc — ML basics

| | |
|---|---|
| **What** | Added revision notes aligned with training walkthrough. |
| **Why** | Single place to revise tensors, CNN, loss, metrics, config. |
| **Files** | `notebooks/ML_BASICS_REVISION.md` |
| **Notes** | Expanded over following days (metrics, MPS, TAT chapter). |

---

## 2026-05-26 — Training speed (Apple Silicon)

### MPS device support

| | |
|---|---|
| **What** | `get_device()` prefers CUDA → **MPS** → CPU. |
| **Why** | Training was ~49 min/epoch on CPU; M1/M2 GPU available but unused. |
| **Files** | `src/utils.py` |
| **Notes** | Verify log shows `Using device: mps`. |

### DataLoader tuning

| | |
|---|---|
| **What** | `persistent_workers`, `prefetch_factor=2`; `pin_memory` only on CUDA. |
| **Why** | Faster epoch time on GPU/MPS. |
| **Files** | `src/train.py`, `configs/config.yaml` (`num_workers: 4`) |

---

## 2026-05-26 — Train & compare all backbones

### `train_all` script

| | |
|---|---|
| **What** | One command trains `efficientnet_b0`, `resnet50`, `mobilenet_v3_large` sequentially. |
| **Why** | User wanted terminal workflow for all three models without three manual commands. |
| **Files** | `src/train_all.py`, `README.md` |
| **Commands** | `python -m src.train_all` · `--skip-existing` · `--set-default efficientnet_b0` |
| **Notes** | **Overwrites** `best_model_<backbone>.pt` unless `--skip-existing`. Each run also overwrites `best_model.pt` (last backbone wins). |

### Per-backbone checkpoints

| | |
|---|---|
| **What** | `train.py` saves `best_model_{backbone}.pt` plus `best_model.pt`. |
| **Why** | Keep all three models for comparison; default file for legacy paths. |
| **Files** | `src/train.py` |

### Model comparison on test set

| | |
|---|---|
| **What** | `compare_models` evaluates each checkpoint with `--tag`; writes comparison CSV/JSON. |
| **Why** | Side-by-side class / gender / color / joint metrics; avoid single `evaluation_test.json` overwrite. |
| **Files** | `src/compare_models.py`, `src/evaluate.py` (`--tag`), `outputs/metrics/model_comparison_test.csv` |
| **Commands** | `python -m src.compare_models` · `python -m src.compare_models --skip-eval` |

### Baseline test results (pre `color: 1.2` retrain)

| Backbone | Class | Gender | Color | Joint |
|----------|-------|--------|-------|-------|
| EfficientNet-B0 | 98.7% | 89.5% | 65.6% | 58.5% |
| ResNet50 | 98.7% | 85.5% | 68.0% | 57.0% |
| MobileNet V3 Large | 98.1% | 86.7% | 64.5% | 55.2% |

**Takeaway:** Class strong; color weakest; EfficientNet best overall joint accuracy.

---

## 2026-05-26 / 2026-05-27 — Streamlit app

### Inference UX

| | |
|---|---|
| **What** | GPU warmup, top-3 alternatives, confidence badges, review queue, test metrics in sidebar. |
| **Why** | Better demo; fair TAT; low-confidence flagging. |
| **Files** | `src/inference.py`, `app/streamlit_app.py`, `src/active_learning.py` |

### Model dropdown (single select)

| | |
|---|---|
| **What** | Replaced checkpoint text input with dropdown of `best_model_*.pt`; sidebar metrics from `evaluation_test_<backbone>.json`. |
| **Why** | Compare three trained models without editing paths. |
| **Files** | `app/streamlit_app.py` (`discover_checkpoints`, `load_test_metrics`) |
| **Notes** | Restart Streamlit after code changes. |

---

## 2026-05-27 — Inference batching & loss tuning (config)

### True batch inference

| | |
|---|---|
| **What** | `predict_batch_pil`: stack N images → one `model()` forward; Streamlit uses one pass for all uploads. |
| **Why** | Lower average TAT in demo; match train/eval batching story. |
| **Files** | `src/inference.py`, `app/streamlit_app.py`, `src/benchmark_latency.py`, `notebooks/ML_BASICS_REVISION.md` (Ch. 12) |
| **Notes** | Each result has `tat_ms` (batch÷N) and `batch_tat_ms` (total forward). `predict_pil` delegates to batch with N=1. |

### Color loss weight increase

| | |
|---|---|
| **What** | `training.loss_weights.color`: **0.8 → 1.2** |
| **Why** | Color head underperforms; encourage model to focus more on color during training. |
| **Files** | `configs/config.yaml` |
| **Notes** | **Requires retrain** — existing `.pt` files still reflect old weights until `train_all` or per-backbone `train` runs. Backup old checkpoints if needed: `outputs/models/backup_*`. |

---

## 2026-05-27 — Usage head (4th task)

| | |
|---|---|
| **What** | Added `usage` from raw CSV: mapping in config, 4th model head, joint accuracy over class+gender+color+usage. |
| **Why** | Bonus attribute; metadata already in Kaggle `usage` column. |
| **Files** | `configs/config.yaml`, `src/label_mapping.py`, `src/data_preparation.py`, `src/model.py`, `src/dataset.py`, `src/train.py`, `src/evaluate.py`, `src/inference.py`, `src/active_learning.py`, `src/compare_models.py`, `src/optimize_onnx.py`, `app/streamlit_app.py`, `src/tasks.py` |
| **Notes** | **Dominant-color fallback cancelled.** Re-run data prep + retrain before usage appears in checkpoints. |

```bash
python -m src.data_preparation
python -m src.train_all --set-default efficientnet_b0
```

---

## Planned (not implemented yet)

Full notes: **[PLANNED_FEATURES.md](./PLANNED_FEATURES.md)**

| Item | Why | Status |
|------|-----|--------|
| **Dominant-color fallback** | Color robustness heuristic | **Cancelled** |
| Staged fine-tune (freeze backbone → unfreeze last blocks → low LR) | Improve gender/color without wrecking pretrained features | Planned |
| Longer training (8–12 epochs, `lr=1e-4`) | Current 5 epochs may be short | Config still `epochs: 5` |
| Streamlit 3-model side-by-side on one image | Compare predictions without switching dropdown | Planned |
| Weight pruning (`torch.prune`) | Smaller/faster model | Not in repo |
| ONNX path wired into Streamlit | Quantized deploy | Export only via `optimize_onnx.py` |
| Real batched ONNX runtime | Production batch server | Future |

---

## Quick command reference

```bash
# Train all three (overwrites checkpoints)
.venv/bin/python -m src.train_all --config configs/config.yaml --set-default efficientnet_b0

# Skip backbones that already have .pt files
.venv/bin/python -m src.train_all --skip-existing

# Test-set comparison table
.venv/bin/python -m src.compare_models --split test

# Latency: single vs true batch
.venv/bin/python -m src.benchmark_latency --checkpoint outputs/models/best_model_efficientnet_b0.pt

# UI
streamlit run app/streamlit_app.py
```

---

## Revision log (this file)

| Date | Entry |
|------|--------|
| 2026-05-27 | Initial timeline from project development sessions |

<!-- Add new rows at the top of the dated sections when you ship work -->
