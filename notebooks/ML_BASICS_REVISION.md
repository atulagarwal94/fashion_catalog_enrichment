# ML & CNN basics — revision notes (living doc)

> Personal revision sheet for [02_training_walkthrough.ipynb](./02_training_walkthrough.ipynb).  
> Update this file as you learn. Last refreshed: 2026-05-27.

---

## How to use this doc

1. Read one **chapter** at a time (matches notebook sections).
2. After running notebook cells, add your own notes under **My notes**.
3. Open the linked `src/` file when a chapter points to it.

---

## The goal in one sentence

We have **product photos** and **correct labels** (type, gender, color). The program learns from many examples so it can **guess those three labels** for a new photo.

---

## Learning plan (notebook order)

| Step | Notebook § | Topic | Code to open |
|------|------------|-------|--------------|
| 1 | 0 | Setup, paths, imports | setup cells |
| 2 | 1 | Config YAML | `configs/config.yaml` |
| 3 | 2 | Data preparation | `src/data_preparation.py` |
| 4 | 3 | Explore `train.csv` | `data/processed/train.csv` |
| 5 | 4 | Image → tensor, Dataset | `src/dataset.py` |
| 6 | 5 | Model (backbone + heads) | `src/model.py` |
| 7 | 6 | Loss, sampler, DataLoader | `src/train.py` |
| 8 | 7 | One training step | notebook §7 |
| 9 | 8–9 | Mini / full training | `src/train.py` |
| 10 | 10–11 | Metrics, inference | `src/inference.py` |
| 11 | — | TAT, batching, quantization, pruning | `src/inference.py`, `src/benchmark_latency.py`, `src/optimize_onnx.py` |

### My notes (progress)

- [ ] Finished section 0–3
- [ ] Finished section 4 (tensors)
- [ ] Finished section 5–7
- [ ] Ran mini training
- [ ] Ran full training

_Add your own checklist or dates here._

---

## Chapter 1 — Digital images and tensors

### What is a digital image?

A photo is a grid of **pixels**. Each pixel has **3 color channels**:

| Channel | Meaning | Typical range (in a JPG) |
|---------|---------|---------------------------|
| R | Red brightness | 0–255 |
| G | Green | 0–255 |
| B | Blue | 0–255 |

```text
Human view:          Computer view (simplified):
┌─────────┐          height × width × 3 numbers
│  photo  │    →     (one number per R/G/B per pixel)
└─────────┘
```

### What is a tensor?

**Layman definition:** A **tensor** is a multi-dimensional **box of numbers** that PyTorch can do math on quickly.

| Type | Dimensions | Example |
|------|------------|---------|
| Scalar | 0D | One number: `3` |
| Vector | 1D | List: `[1, 0, 2]` |
| Matrix | 2D | Spreadsheet |
| Image tensor | 3D | **3 sheets** (R, G, B) × height × width |

**“Convert image to tensor”** = turn the JPG into that standardized number box so the model can use it. Like exporting a file to a format your app understands.

### Shape after our transforms

One image in this project:

```text
[3, 224, 224]

 3   = RGB channels
224  = height (pixels)
224  = width (pixels)
```

A **batch** (many images at once):

```text
[32, 3, 224, 224]
 ↑
 batch_size from config.yaml
```

### My notes — images & tensors

<!-- Add questions, ah-ha moments, or things you forget -->

---

## Chapter 2 — What happens to one image in code

File: `src/dataset.py` → `FashionCatalogDataset` + `build_transforms`

| Step | What code does | Why (plain English) |
|------|----------------|---------------------|
| 1 | `Image.open(path)` | Open the JPG |
| 2 | `.convert("RGB")` | Always 3 color channels |
| 3 | `Resize((224, 224))` | Same size for every photo |
| 4 | Flip / rotate / color jitter (train only) | Slight variations so model doesn’t memorize exact pixels |
| 5 | `ToTensor()` | Pixels 0–255 → floats 0.0–1.0; shape becomes `[3, H, W]` |
| 6 | `Normalize(mean, std)` | Match colors to how the pretrained backbone was trained |

### ToTensor (simple)

- Before: integers 0–255 per channel.
- After: floats 0.0–1.0, channels first: `[3, height, width]`.

### Normalize (simple)

The backbone was pretrained on ImageNet with a fixed color “recipe.” Normalization applies the same recipe:

```text
new_value = (old_value - mean) / std   (per channel)
```

So values are often roughly between about -2 and +2, not 0–255.

### Targets (the answer key)

Each row in `train.csv` also gives **three small integers**:

| Column | Example | Meaning |
|--------|---------|---------|
| `class_idx` | `3` | e.g. Shirt (see `labels.json`) |
| `gender_idx` | `1` | e.g. Men |
| `color_idx` | `2` | e.g. Blue |

Words live in `labels.json`; training uses **numbers** because loss functions compare numbers.

### Hands-on checks (run in notebook)

```python
sample = train_ds[0]
print(sample["image"].shape)       # torch.Size([3, 224, 224])
print(sample["image"].min(), sample["image"].max())
print(sample["targets"])
print(label_info["class"]["idx_to_label"][str(int(sample["targets"]["class"]))])
```

### My notes — transforms & dataset

---

## Chapter 3 — What is a CNN? (no jargon)

**CNN** = **Convolutional Neural Network** = software built to **find patterns in images**.

### Analogy

| Layer (conceptual) | Looks for |
|--------------------|-----------|
| Early | Edges, blobs |
| Middle | Parts (sleeve, sole, strap) |
| Deep | Whole object type |

We don’t write these rules by hand. **Training** adjusts internal **weights** until predictions match labels.

### Backbone vs heads (this project)

```text
Photo → BACKBONE (shared vision expert) → summary vector
                    ↓
         ┌──────────┼──────────┐
    class head   gender head   color head
    (Shirt?)     (Men?)        (Blue?)
```

- **Backbone:** EfficientNet / ResNet / MobileNet — pretrained on many images.
- **Heads:** Small layers that each answer one question.

**Why one backbone?** One look at the photo; cheaper and often better than three separate models.

### My notes — CNN / model

---

## Chapter 4 — Data pipeline (before training)

### Config (`configs/config.yaml`)

Single **settings menu**: paths, image size, batch size, epochs, which labels to keep, train/val split sizes.

### Data preparation (`src/data_preparation.py`)

| Input | Output |
|-------|--------|
| `data/raw/styles.csv` | `data/processed/train.csv`, `val.csv`, `test.csv` |
| `data/raw/images/{id}.jpg` | `labels.json` (word ↔ index maps) |

**In plain steps:**

1. Read Kaggle metadata.
2. Map raw names → clean labels (`src/label_mapping.py`).
3. Drop rows with missing/broken images or too few examples per class.
4. Assign integer indices.
5. Split into train / validation / test.

### My notes — data prep

---

## Chapter 5 — Training loop (story)

```text
1. DataLoader brings a BATCH of images + true labels
2. MODEL forward → predictions (scores per class)
3. LOSS → one number: how wrong we are
4. backward() → which weights to nudge
5. optimizer.step() → nudge weights a tiny bit
6. Repeat for all batches → one EPOCH
7. Repeat epochs; check VALIDATION set; save best model
```

| Term | Plain meaning |
|------|----------------|
| **Batch** | Group trained together (e.g. 32 photos) |
| **Epoch** | One full pass over all training rows |
| **Loss** | Wrongness score (lower is better) |
| **Validation** | Held-out photos; tests generalization |
| **Checkpoint** | Saved weights (`outputs/models/best_model.pt`) |

### Weighted sampler & weighted loss

**Problem:** Many more Shirts than Caps in data → model ignores rare classes.

**Fix:**

- **Sampler:** Rare classes appear more often in batches.
- **Weighted loss:** Mistakes on rare classes hurt more.

### Combined loss (three tasks)

```text
total = w_class × loss_class + w_gender × loss_gender + w_color × loss_color
```

Weights from `config.yaml` → `training.loss_weights`.

### My notes — training

---

## Chapter 5b — Mini-training metrics (how to read the scorecard)

Example output:

```text
Mini train loss: 4.6466
Validation metrics: {
  'loss': 4.0289,
  'class_accuracy': 0.7616,
  'class_macro_f1': 0.7075,
  'gender_accuracy': 0.5682,
  'gender_macro_f1': 0.4387,
  'color_accuracy': 0.1908,
  'color_macro_f1': 0.1174,
  'joint_accuracy': 0.0684
}
```

| Metric | Plain meaning | Your example (≈) |
|--------|----------------|------------------|
| **Train loss** | Average “wrongness” on the **512 images you trained on** just now | 4.65 — still high after 1 quick epoch |
| **Val loss (`loss`)** | Same loss formula on **validation** images (not in mini-train subset) | 4.03 — reference for generalization |
| **`class_accuracy`** | % of val images where **product type** is exactly right | **76%** — best task so far |
| **`gender_accuracy`** | % where **gender** is right | **57%** |
| **`color_accuracy`** | % where **color** is right | **19%** — hardest here (15 color labels) |
| **`*_macro_f1`** | Score that treats **each label equally** (good for imbalanced classes) | Class F1 **71%** > gender **44%** > color **12%** |
| **`joint_accuracy`** | % where **class AND gender AND color** are all right on the **same** photo | **6.8%** — strict; \(0.76 × 0.57 × 0.19 ≈ 8\%\) ballpark |

**Accuracy vs macro F1:** If the model always predicts “Shirt,” accuracy can look okay when Shirts are common; macro F1 punishes ignoring rare labels.

**Why joint is so low:** You must win three quizzes on one image. Even decent per-task scores multiply to a small number.

**After mini-training only:** Do not expect production quality. Use charts in notebook **§8b** (`plot_mini_training_metrics`) or full training + history plots (§10).

### Train loss / val loss — scale and how to judge 4.65 vs 4.03

**Not a percentage.** Loss is an open-ended penalty score (lower = better). There is no fixed “100% loss.”

**What you add up** (`src/train.py` + `config.yaml`):

```text
total_loss = 1.0 × loss_class + 0.7 × loss_gender + 0.8 × loss_color
```

Each piece is **cross-entropy** for one head (how badly wrong the class scores are).

**Reference: random guessing** (untrained model, roughly uniform guesses):

| Task | # labels | Random CE ≈ ln(labels) | × weight |
|------|----------|-------------------------|----------|
| class | 8 | ~2.08 | × 1.0 |
| gender | 4 | ~1.39 | × 0.7 |
| color | 15 | ~2.71 | × 0.8 |
| **Sum** | | | **~5.2** |

So **~5.2** ≈ “coin-flip level” for all three tasks. Your **4.65 / 4.03** are **a bit better than random** — consistent with class ~76% but color still weak.

**Anchors:**

| Total loss (ballpark) | Meaning |
|-----------------------|---------|
| **~5+** | Near random / barely trained |
| **~3–4** | Learning; mini-run territory |
| **~1–2** | Decent after real training (depends on data) |
| **→ 0** | Perfect (never exactly 0 in practice) |

**How to evaluate 4.65 vs 4.03:**

1. **Direction over time** — loss should **drop** across epochs (main use in training).
2. **Train vs val** — val **4.03** &lt; train **4.65** here → no overfitting on this tiny run; gap may grow if model memorizes.
3. **Use accuracy/F1 for “how good”** — loss is for the optimizer; humans read **76% class**, **7% joint**, etc.
4. **Compare to ~5.2 random baseline**, not to 0.

**Per-image intuition:** CE ≈ `-log(probability on true label)`. Confident and wrong → huge loss; confident and right → small loss.

### My notes — metrics

---

## Chapter 6 — Inference

Load `best_model.pt` → same transforms as validation (no random flip) → print predicted class, gender, color.

File: `src/inference.py`

### My notes — inference

---

## End-to-end diagram

```text
  JPG on disk
      │
      ▼  open, resize, ToTensor, Normalize
  Tensor [3, 224, 224]
      │
      ▼
  CNN backbone → feature vector
      │
      ├── class head  → scores → pick label
      ├── gender head → scores → pick label
      └── color head  → scores → pick label
      │
  Compare to CSV truth (class_idx, gender_idx, color_idx)
      │
      ▼
  Loss → update weights → repeat
```

---

## Chapter 12 — TAT, batch efficiency, quantization, pruning (and where they live in code)

> **Kid version:** TAT = stopwatch for one answer. Batch = school bus for many photos. Quantization = simpler numbers in the backpack. Pruning = cutting unused wires (we mostly use a *smaller bus* instead).

### How the four ideas connect

```text
  Upload / folder of images
           │
           ├─► [Batch efficiency]  train & eval use batch_size=32 on GPU
           │                       Streamlit uses 1 image → higher TAT per photo
           │
           ├─► [TAT measured]      time around model.forward in inference.py
           │
           ├─► [Quantization]      optional ONNX INT8 export (optimize_onnx.py)
           │                       not wired into Streamlit yet
           │
           └─► [Pruning]           NOT implemented in this repo yet
                                   we use smaller backbones (MobileNet) instead
```

---

### 1. TAT (Turnaround Time) — **implemented**

**Plain meaning:** Milliseconds from “start forward pass” to “have class + gender + color.”

**Kid analogy:** Time from pressing “Go” on the quiz until all three answers are on the paper.

| Where | What it does |
|-------|----------------|
| `src/inference.py` → `predict_pil()` | `time.perf_counter()` before/after `model(tensor)` → `tat_ms` in JSON |
| `src/inference.py` → `_warmup()` | One dummy forward on load so **first** Streamlit image is not unfairly slow |
| `app/streamlit_app.py` | Shows per-image TAT + **Avg TAT** across uploads |
| `src/benchmark_latency.py` | Mean/median/min/max/p95 TAT over many files → `latency_benchmark.json` |
| `src/active_learning.py` | Saves `tat_ms` in review-queue CSV |

**Code path (single upload):**

```text
PIL image
  → transform → tensor shape [1, 3, 224, 224]   # batch dim = 1
  → start timer
  → model(tensor)  → three heads
  → stop timer → tat_ms
```

**Run from terminal:**

```bash
python -m src.inference --checkpoint outputs/models/best_model_efficientnet_b0.pt --image path/to.jpg
python -m src.benchmark_latency --checkpoint outputs/models/best_model_efficientnet_b0.pt --image_dir data/sample_images
```

**What TAT does *not* include today:** disk read, JPEG decode, or Streamlit UI paint — only the neural net forward pass (plus tensor prep right before it).

---

### 2. Batch efficiency — **partly implemented**

**Plain meaning:** Processing **many images in one GPU trip** is cheaper **per image** than one trip per image.

**Kid analogy:** One bus for 32 kids vs 32 buses for 32 kids.

#### Where we **do** use real batches (good efficiency)

| File | Setting | Code idea |
|------|---------|-----------|
| `configs/config.yaml` | `training.batch_size: 32` | How many images per training step |
| `src/train.py` | `make_loader(..., batch_size=32)` | `for batch in dataloader:` → `images = batch["image"]` shape `[32, 3, 224, 224]` |
| `src/evaluate.py` | Same `batch_size` from config | Test-set eval in batches of 32 |

One `model(images)` call = one forward for 32 photos → **training/eval batch efficiency**.

#### True inference batch (Streamlit + API) — **implemented**

| File | Behavior |
|------|----------|
| `src/inference.py` → `predict_batch_pil()` | Stack N images → `[N, 3, 224, 224]` → **one** `model(batch)` → decode each row |
| `src/inference.py` → `predict_pil()` | Calls `predict_batch_pil` with N=1 |
| `app/streamlit_app.py` | All uploads in **one** forward pass |
| `src/benchmark_latency.py` | Reports `batch_forward_ms` and `batch_avg_ms_per_image` |

Each result includes `tat_ms` (batch time ÷ N) and `batch_tat_ms` (total forward pass).

**ONNX export already allows variable batch dim** (`dynamic_axes` in `src/optimize_onnx.py`) so a batched inference server could use `N > 1` later.

---

### 3. Quantization — **implemented (export path only)**

**Plain meaning:** Store weights with **smaller integers** (e.g. INT8) instead of big floats → smaller file, often faster on CPU.

**Kid analogy:** Cheat sheet with rounded numbers instead of a huge textbook.

| Where | What |
|-------|------|
| `configs/config.yaml` → `optimization.quantize_dynamic: true` | Turn on/off INT8 step |
| `configs/config.yaml` → `optimization.onnx_opset: 17` | ONNX version for export |
| `src/optimize_onnx.py` | 1) Export PyTorch → `model.onnx` 2) `quantize_dynamic(..., QInt8)` → `model_quantized.onnx` |
| `src/model.py` → `ONNXExportWrapper` | Flattens three heads for ONNX export |
| `outputs/metrics/onnx_optimization_report.json` | File sizes + status |

**Run:**

```bash
python -m src.optimize_onnx \
  --checkpoint outputs/models/best_model_efficientnet_b0.pt \
  --output_dir outputs/models
```

**Not wired yet:** Streamlit and `FashionCatalogPredictor` still load **`.pt` PyTorch** checkpoints, not ONNX. Quantization is a **deployment prep** step, not what you see in the live app today.

---

### 4. Pruning — **not implemented (related things we do have)**

**Plain meaning:** Remove weak connections/neurons after training so the network is thinner and faster.

**Kid analogy:** Cutting dead branches off a tree so less to climb.

| In this repo | Status |
|--------------|--------|
| `torch.nn.utils.prune` / structured pruning | **Not in codebase** |
| `dropout` in `src/model.py` (`shared_dropout`) | **Training regularization** — randomly drops activations while learning; **not** the same as pruning |
| Picking **MobileNet** vs **ResNet** in `src/train_all.py` | **Smaller architecture** — fewer parameters by design (like choosing a smaller bus, not pruning a big one) |

If you add pruning later, typical place: new script `src/prune_model.py` → save `best_model_*_pruned.pt` → re-benchmark with `benchmark_latency.py`.

---

### Quick map: term → file → command

| Term | Main file(s) | Try it |
|------|----------------|--------|
| **TAT** | `src/inference.py`, `app/streamlit_app.py`, `src/benchmark_latency.py` | `streamlit run app/streamlit_app.py` |
| **Batch (train/eval)** | `src/train.py`, `src/evaluate.py`, `configs/config.yaml` | `python -m src.train --backbone efficientnet_b0` |
| **Batch (inference)** | `src/inference.py` → `predict_batch_pil` | Upload N images in Streamlit |
| **Quantization** | `src/optimize_onnx.py`, `configs/config.yaml` | `python -m src.optimize_onnx` |
| **Pruning** | — | Not implemented |

### Compare backbones for speed vs accuracy

Same code path, different checkpoint size / speed:

| Checkpoint | Role |
|------------|------|
| `best_model_efficientnet_b0.pt` | Best accuracy on our test summary |
| `best_model_mobilenet_v3_large.pt` | Smaller/faster backbone |
| `best_model_resnet50.pt` | Largest file, mixed trade-offs |

```bash
python -m src.benchmark_latency --checkpoint outputs/models/best_model_mobilenet_v3_large.pt
python -m src.compare_models --skip-eval   # accuracy side-by-side
```

### My notes — speed & deployment

<!-- Add your measured TAT numbers, ONNX sizes, ideas for true batched inference -->

---

## Glossary

| Word | Plain meaning |
|------|----------------|
| Tensor | Box of numbers for math |
| Channel | One color layer (R, G, or B) |
| Batch | Several images processed together |
| Transform | Resize / flip / tensor / normalize |
| Label / target | Correct answer for training |
| idx | Answer as integer (3 = Shirt) |
| Logits | Raw scores before choosing winner |
| Loss | How wrong the model is |
| Gradient / backward | Directions to improve weights |
| Optimizer | Applies small weight updates |
| Epoch | One lap through all training data |
| Backbone | Image feature extractor |
| Head | Classifier for one task |
| Pretrained | Already learned general vision elsewhere |
| Checkpoint | Saved model file |
| TAT | Turnaround time (ms) for one prediction forward pass |
| Batch efficiency | Lower ms per image when many images share one `model()` call |
| Quantization | Smaller numeric weights (e.g. INT8) for faster/smaller deploy |
| Pruning | Remove weak weights after training (not in this repo yet) |
| Dropout | Random off-switch during training only (not pruning) |
| ONNX | Exported model format for deployment runtimes |

---

## Notebook section → file map

| Notebook § | Source files |
|------------|----------------|
| Config | `configs/config.yaml`, `src/utils.py` |
| Data prep | `src/data_preparation.py`, `src/label_mapping.py` |
| Dataset | `src/dataset.py` |
| Model | `src/model.py` |
| Train / eval | `src/train.py` |
| Inference | `src/inference.py` |
| TAT benchmark | `src/benchmark_latency.py` |
| ONNX + quantization | `src/optimize_onnx.py` |
| Streamlit UI | `app/streamlit_app.py` |

---

## Open questions / to research later

<!-- Add bullets as you study; move answers here when resolved -->

- 
- 

---

## Revision log

| Date | What I added or clarified |
|------|---------------------------|
| 2026-05-26 | Initial basics from walkthrough explanation |
| 2026-05-27 | Chapter 12: TAT, batching, quantization, pruning linked to `src/` |
