# ML & CNN basics — revision notes (living doc)

> Personal revision sheet for [02_training_walkthrough.ipynb](./02_training_walkthrough.ipynb).  
> Update this file as you learn. Last refreshed: 2026-05-30.

---

## How to use this doc

1. Read one **chapter** at a time (matches notebook sections).
2. After running notebook cells, add your own notes under **My notes**.
3. Open the linked `src/` file when a chapter points to it.

---

## The goal in one sentence

We have **product photos** and **correct labels** (type, gender, color, usage). The program learns from many examples so it can **guess those four labels** for a new photo.

---

## Learning plan (notebook order)

| Step | Notebook § | Topic | Code to open |
|------|------------|-------|----|
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

**"Convert image to tensor"** = turn the JPG into that standardized number box so the model can use it. Like exporting a file to a format your app understands.

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
| 4 | Flip / rotate / color jitter (train only) | Slight variations so model doesn't memorize exact pixels |
| 5 | `ToTensor()` | Pixels 0–255 → floats 0.0–1.0; shape becomes `[3, H, W]` |
| 6 | `Normalize(mean, std)` | Match colors to how the pretrained backbone was trained |

### ToTensor (simple)

- Before: integers 0–255 per channel.
- After: floats 0.0–1.0, channels first: `[3, height, width]`.

### Normalize (simple)

The backbone was pretrained on ImageNet with a fixed color "recipe." Normalization applies the same recipe:

```text
new_value = (old_value - mean) / std   (per channel)
```

So values are often roughly between about -2 and +2, not 0–255.

### Why normalization matters — a deeper look

Imagine you're comparing two people's heights, but one is measured in centimeters and the other in inches. You'd convert them to the same unit first. Normalization does the same for pixel values.

The pretrained backbone (MobileNetV3, ResNet50, etc.) was trained on ImageNet with specific mean and std values. If you feed it images with a different scale, the features it extracts will be wrong — like asking someone who reads Celsius to interpret Fahrenheit readings.

```text
ImageNet stats (used by ALL common backbones):
  mean = [0.485, 0.456, 0.406]   ← average R, G, B across 14M images
  std  = [0.229, 0.224, 0.225]   ← spread of R, G, B

These are NOT arbitrary — they're computed from the actual ImageNet dataset.
```

### Data augmentation — why each transform exists

| Transform | What it does | Why it helps YOUR project |
|-----------|-------------|--------------------------|
| RandomHorizontalFlip(p=0.5) | Mirror image left-right | A blue shirt flipped is still a blue shirt. Prevents model from learning "shirts are always facing right" |
| RandomRotation(10°) | Slight tilt | Real products aren't always perfectly straight. Catalog photos can be slightly rotated |
| ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) | Randomly shift lighting/colors | **Critical for color prediction.** The same "blue" shirt looks different under warm vs cool lighting. Forces model to learn color robustly |
| Resize(224, 224) | Standard size | Backbone expects exactly this size. Different sized images would break the math |
| Normalize(mean, std) | Shift pixel range | Match pretrained backbone's expected input distribution |

**Why augmentation is training-only:** During validation and testing, you want a clean, consistent measurement. If you augment test images, your metrics would be noisy — like taking an exam with someone randomly spinning the paper.

### Targets (the answer key)

Each row in `train.csv` also gives **four small integers**:

| Column | Example | Meaning |
|--------|---------|---------|
| `class_idx` | `3` | e.g. Shirt (see `labels.json`) |
| `gender_idx` | `1` | e.g. Men |
| `color_idx` | `2` | e.g. Blue |
| `usage_idx` | `0` | e.g. Casual |

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

We don't write these rules by hand. **Training** adjusts internal **weights** until predictions match labels.

### What is a convolution?

Imagine a small magnifying glass (3×3 pixels) sliding across your photo. At each position, it multiplies each pixel by a number in the magnifying glass, sums them up, and writes one number in the output.

```text
Input image          Filter (3×3)         Output (one number)
┌───┬───┬───┐       ┌────┬────┬────┐
│ 1 │ 2 │ 3 │       │ -1 │  0 │  1 │     sum of element-wise
│ 4 │ 5 │ 6 │   ×   │ -1 │  0 │  1 │  =  multiply = 6
│ 7 │ 8 │ 9 │       │ -1 │  0 │  1 │     (this filter detects vertical edges)
└───┴───┴───┘       └────┴────┴────┘
```

The model learns WHICH filters to use. Early filters find edges, later filters find complex shapes.

### What is pooling?

After convolution, **pooling** shrinks the output. Max pooling takes the maximum value in each 2×2 region:

```text
┌───┬───┐        ┌───┐
│ 4 │ 6 │   →    │ 9 │   (took the max from each 2×2 block)
│ 7 │ 9 │        └───┘
└───┴───┘
```

Why? Reduces computation and makes the model slightly tolerant to position shifts (a shoe moved 2 pixels left is still a shoe).

### What is Global Average Pooling?

At the end of the backbone, you have (say) 1280 feature maps, each a small grid. Global Average Pooling takes the **average of each entire grid**, giving you a single vector of 1280 numbers. This is the "embedding" — a compressed summary of everything the model sees in the image.

```text
1280 feature maps (each 7×7)  →  Global Avg Pool  →  [1280] vector
                                                       ↑
                                                  This is the embedding
```

### Backbone vs heads (this project)

```text
Photo → BACKBONE (shared vision expert) → summary vector (embedding)
                    ↓
         ┌──────────┼──────────┬──────────┐
    class head   gender head  color head  usage head
    (Shirt?)     (Men?)       (Blue?)     (Casual?)
```

- **Backbone:** EfficientNet / ResNet / MobileNet — pretrained on many images.
- **Heads:** Small layers that each answer one question.

**Why one backbone?** One look at the photo; cheaper and often better than three separate models.

### What is a Linear layer (the head)?

Each head is literally one matrix multiplication:

```python
class_head = nn.Linear(1280, 8)  # 1280 inputs → 8 outputs (one per class)
```

This takes the 1280-number embedding and produces 8 scores (one per product category). The highest score wins.

```text
Embedding [1280 numbers]
    ↓
    × weight matrix (1280 × 8) + bias (8)
    ↓
Raw scores [8 numbers]: [2.1, 0.3, -1.5, 5.8, 0.1, -0.7, 1.2, 0.4]
    ↓
Softmax → probabilities: [0.05, 0.01, 0.00, 0.89, 0.01, 0.00, 0.03, 0.01]
    ↓
Pick highest: index 3 = "Shoes" with 89% confidence
```

### Why multi-head over separate models?

| Approach | Forward passes | Memory | Consistency |
|----------|---------------|--------|-------------|
| 4 separate models | 4 (one per model) | 4x model weights in RAM | Each model sees image differently |
| 1 shared backbone + 4 heads | 1 (shared) | 1x backbone + tiny heads | All predictions from same understanding |

**The shared backbone captures general visual features.** A shoe's color and its category come from the same visual information — no need to look at the image 4 times.

### The three backbones compared — architecture intuition

**ResNet50 (2015) — "The reliable workhorse"**

Innovation: skip connections (residual connections). In a normal deep network, gradients vanish as they travel backward through many layers. ResNet adds shortcuts that let gradients skip layers:

```text
Normal:    x → Layer1 → Layer2 → output
ResNet:    x → Layer1 → Layer2 → output + x   ← skip connection
```

This lets you train much deeper networks (50, 101, 152 layers) without the gradient vanishing problem.

- 50 layers, ~25M parameters
- Largest model in your project
- Highest class accuracy but slowest inference

**EfficientNet-B0 (2019) — "The AI-designed architecture"**

Innovation: compound scaling. Instead of a human deciding how deep/wide to make the network, a computer search found the optimal balance of depth (layers), width (channels), and resolution (input size).

- ~5.3M parameters (5x smaller than ResNet50)
- Good balance of accuracy and speed
- Best gender accuracy in your project

**MobileNetV3 Large (2019) — "The speed demon"**

Innovation: depthwise separable convolutions. A normal convolution is one big operation. MobileNet splits it into two smaller operations that produce the same result with far fewer calculations:

```text
Normal conv:       1 operation,  N × N × C_in × C_out  multiplications
Depthwise sep.:    2 operations, N × N × C_in + C_in × C_out  multiplications
                   (usually 8-9x fewer multiplications!)
```

Also uses squeeze-and-excitation blocks (channel attention) and h-swish activation.

- ~5.4M parameters
- Designed for mobile phones — very fast
- YOUR WINNER: best color, best joint accuracy, fastest on EC2

**Why did the smallest model win?**

Fashion products have relatively simple visual features compared to (say) medical images or satellite imagery. The features that distinguish a shoe from a watch — shape, outline, typical colors — are captured well by MobileNetV3's efficient architecture. ResNet50's extra capacity gives it more power to model subtle features, but that extra power isn't needed here, and it can even hurt by making training noisier on the limited dataset.

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

### Label mapping — why and how

The raw Kaggle data has messy labels:

```text
Raw labels:                     Mapped to:
"Tshirts"                  →   "T-Shirt"
"Casual Shoes"              →   "Shoes"
"Sports Shoes"              →   "Shoes"
"Formal Shoes"              →   "Shoes"
"Jeans"                     →   "Pants"
"Trousers"                  →   "Pants"
```

Why group them? If you kept "Casual Shoes," "Sports Shoes," and "Formal Shoes" as separate classes, you'd need more data per class and the model would need to learn finer distinctions. For V1, "Shoes" is sufficient.

### Color grouping

Same idea for colors:

```text
Raw:                           Mapped to:
"Navy Blue"                →   "Blue"
"Maroon"                   →   "Red"
"Teal"                     →   "Green"
"Beige", "Cream"           →   "Other"
"Lavender", "Mauve"        →   "Other"

Final colors: Black, Blue, White, Red, Green, Grey, Brown, Pink, Yellow, Other
```

Why? Rare colors with few images make training unstable. Grouping ensures every color class has enough examples. "Other" is a catch-all for very rare colors.

### Stratified split — why it matters

```text
BAD random split:
  Train: 8000 shirts, 500 hats, 0 watches  ← model never sees watches!
  Test:  0 shirts, 0 hats, 200 watches     ← can't evaluate shirts!

GOOD stratified split:
  Train: 5600 shirts, 350 hats, 140 watches  (70% of each)
  Val:   1200 shirts,  75 hats,  30 watches  (15% of each)
  Test:  1200 shirts,  75 hats,  30 watches  (15% of each)
```

Every class appears proportionally in every split. Uses `sklearn.model_selection.train_test_split` with `stratify=y`.

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

See Chapter 5c for full details on class imbalance.

### Combined loss (four tasks)

```text
total = w_class × loss_class + w_gender × loss_gender + w_color × loss_color + w_usage × loss_usage
```

Weights from `config.yaml` → `training.loss_weights`.

### My notes — training

---

## Chapter 5a — Loss Functions (deep dive)

> **Kid version:** The loss function is the teacher grading the model's quiz. It gives a number: 0 = perfect answers, big number = terrible answers. Training adjusts the model's brain to get lower grades (lower loss).

### What is a loss function?

A loss function takes two things:
1. **What the model predicted** (probabilities for each label)
2. **What the correct answer was** (the true label)

And outputs **one number** saying how wrong the model was. The entire training process is about making this number smaller.

### Cross-Entropy Loss — the one we use

Cross-Entropy (CE) is the standard loss for classification problems. Here's how it works intuitively:

**Step 1:** The model outputs raw scores (called **logits**) for each possible label:

```text
Color head output (logits): [2.1, 0.5, -1.0, 3.8, 0.2, ...]
                             ↑                   ↑
                           Black               Blue
```

**Step 2:** **Softmax** converts logits to probabilities (all positive, sum to 1.0):

```text
Softmax formula:  probability(i) = e^(logit_i) / sum(e^(all logits))

After softmax:  [0.08, 0.02, 0.00, 0.85, 0.01, ...]
                                      ↑
                                    Blue = 85%
```

**Step 3:** Cross-Entropy looks at the probability assigned to the **correct** label:

```text
True label: Blue (index 3)
Model gave Blue: 0.85 probability

CE Loss = -log(0.85) = 0.163   ← small loss, model is confident and right
```

**The key insight:** CE = `-log(probability of the correct class)`

```text
If model gives 95% to correct class → loss = -log(0.95) = 0.05  (very small)
If model gives 50% to correct class → loss = -log(0.50) = 0.69  (medium)
If model gives 5% to correct class  → loss = -log(0.05) = 3.00  (huge!)
If model gives 1% to correct class  → loss = -log(0.01) = 4.60  (catastrophic!)
```

**Why logarithm?** It creates an asymmetric penalty:
- Being confidently RIGHT is barely rewarded (0.95 → 0.05 loss)
- Being confidently WRONG is severely punished (0.05 → 3.0 loss)
- This pushes the model away from confident mistakes, which is exactly what we want

### Visualizing Cross-Entropy

```text
Loss
  5 │ ×
    │  ×
  4 │   ×
    │    ×
  3 │     ×                ← model says 5% chance for correct class
    │      ×
  2 │        ×
    │          ×
  1 │             ×
    │                 ×
  0 │──────────────────×── ← model says 95%+ for correct class
    └──────────────────────
    0%   20%  40%  60%  80% 100%
       Probability on correct class
```

### Class-weighted Cross-Entropy — handling imbalance

Standard CE treats all classes equally. But if you have 8000 shirts and 500 hats, the model sees 16x more shirt examples. Without correction, it learns "when in doubt, guess Shirt."

**Class-weighted CE** multiplies the loss by a weight per class:

```text
Standard CE:  Loss = -log(P_correct)
Weighted CE:  Loss = weight[true_class] × -log(P_correct)

weight[Shirts] = 1/8000 (normalized)  ← small multiplier (common class)
weight[Caps]   = 1/500  (normalized)  ← large multiplier (rare class)
```

So getting a Cap wrong produces ~16x more loss than getting a Shirt wrong. This forces the model to pay attention to rare classes.

### Your project's combined loss — multi-task weighted loss

Your model has **four heads**, each with its own CE loss. They're combined with **task weights**:

```text
Total Loss = 1.0 × CE_class
           + 0.7 × CE_gender
           + 1.2 × CE_color     ← HIGHEST weight
           + 0.6 × CE_usage     ← LOWEST weight
```

**In code** (`src/train.py`):

```python
loss_class  = criterion_class(class_logits, class_targets)    # CE with class weights
loss_gender = criterion_gender(gender_logits, gender_targets)  # CE with gender weights
loss_color  = criterion_color(color_logits, color_targets)     # CE with color weights
loss_usage  = criterion_usage(usage_logits, usage_targets)     # CE with usage weights

total_loss = (1.0 * loss_class
            + 0.7 * loss_gender
            + 1.2 * loss_color
            + 0.6 * loss_usage)
```

### Why these specific task weights?

| Head | Weight | Reasoning |
|------|--------|-----------|
| Class | 1.0 | Baseline reference. Class is the primary task. |
| Gender | 0.7 | Gender is relatively easy to learn from visual features (men's vs women's clothing has distinct patterns). Lower weight prevents it from dominating gradients. |
| Color | 1.2 | **Hardest task.** Color is affected by lighting, compression, background, pixelation, multi-color products. Needs extra gradient signal to learn properly. |
| Usage | 0.6 | Usage (Casual, Sports, Formal, etc.) is a secondary attribute. Lower weight because it's a bonus head and we don't want it interfering with the mandatory outputs. |

### Two levels of weighting — don't confuse them

Your training uses **two separate weighting mechanisms** that serve different purposes:

```text
Level 1: CLASS WEIGHTS (inside each CE loss)
  → Handles imbalanced data WITHIN one task
  → "There are 8000 Black items but only 500 Pink items"
  → Makes the model pay attention to rare labels
  → Computed from label frequency: weight = 1/count (normalized)

Level 2: TASK WEIGHTS (combining the four losses)
  → Balances importance ACROSS tasks
  → "Color is harder than gender, give it more gradient signal"
  → Set manually in config.yaml
  → 1.0 / 0.7 / 1.2 / 0.6
```

Both are applied simultaneously:

```text
total_loss = 1.0 × WeightedCE_class(...)    ← CE has class weights inside
           + 0.7 × WeightedCE_gender(...)
           + 1.2 × WeightedCE_color(...)    ← CE has color class weights inside
           + 0.6 × WeightedCE_usage(...)       AND task weight = 1.2 outside
```

### What does the optimizer do with the loss?

```text
Total loss = 3.5 (example)
    ↓
loss.backward()
    → PyTorch traces BACKWARD through every operation
    → Computes: "if I change weight #47382 by +0.001,
                 the loss changes by -0.003" (this is the GRADIENT)
    → Does this for EVERY weight in the model (~5 million for MobileNetV3)
    ↓
optimizer.step()
    → Nudges each weight in the direction that reduces loss
    → Amount of nudge = learning_rate × gradient
    → Adam optimizer adapts the nudge size per weight
```

### What happens when task weights are wrong?

| Problem | Symptom | What happened |
|---------|---------|---------------|
| Color weight too low (e.g. 0.3) | Color accuracy stays at ~20%, class is 98% | Class loss dominates; backbone learns features for class, ignores color-relevant features |
| Color weight too high (e.g. 3.0) | Color improves but class accuracy drops to 90% | Color loss dominates; backbone warps features toward color at the expense of shape/structure |
| All weights equal (1.0) | Color underperforms because it's inherently harder | Easy tasks (class) get good gradients naturally; hard tasks (color) need the extra boost |

**Your finding:** 1.2 for color was the sweet spot. Going higher started hurting class accuracy. This ceiling exists because the backbone has shared features — push too hard on one task and you distort the features other tasks rely on.

### Loss vs accuracy — which to watch?

| Metric | Who uses it | What it tells you |
|--------|------------|-------------------|
| Loss | The optimizer (the machine) | "How wrong am I, mathematically?" — continuous, differentiable |
| Accuracy | You (the human) | "What percentage did I get right?" — discrete, intuitive |
| Macro F1 | You (for imbalanced data) | "How well am I doing across ALL classes equally?" |

You train by minimizing loss. You evaluate by reading accuracy and F1. They usually correlate, but not always — you can have a model with slightly higher loss but better accuracy if the loss is concentrated on ambiguous edge cases.

### Reference: random-guessing loss for your project

An untrained model guesses uniformly, so CE = ln(number of classes):

| Task | # labels | Random CE ≈ ln(labels) | × task weight | Contribution |
|------|----------|------------------------|---------------|-------------|
| Class | 8 | 2.08 | × 1.0 | 2.08 |
| Gender | 4–5 | 1.39–1.61 | × 0.7 | 0.97–1.13 |
| Color | 10 | 2.30 | × 1.2 | 2.76 |
| Usage | 7 | 1.95 | × 0.6 | 1.17 |
| **Total** | | | | **~7.0** |

So ~7.0 = "completely random for all four tasks." Your trained model gets to ~1–2 total loss, meaning it's very confident on most predictions.

### Other loss functions (for context — we don't use these)

| Loss function | Used for | Why we don't use it |
|---------------|----------|---------------------|
| Mean Squared Error (MSE) | Regression (predicting a number like temperature) | We're classifying categories, not predicting continuous values |
| Binary Cross-Entropy (BCE) | Binary classification (yes/no) or multi-label (can be both shirt AND blue) | Our tasks are multi-class: one answer per head, not multiple |
| Focal Loss | When class imbalance is extreme | We handle imbalance with WeightedRandomSampler + class-weighted CE instead. Focal loss down-weights easy examples; it's an alternative approach |
| Triplet Loss | Learning embeddings / similarity | Used in metric learning (e.g., "find similar products"); not for direct classification |
| Label Smoothing CE | When you want softer targets | Turns [0, 0, 1, 0] into [0.02, 0.02, 0.94, 0.02]. Prevents overconfidence. Could be a future experiment |

### My notes — loss functions

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
| **Train loss** | Average "wrongness" on the **512 images you trained on** just now | 4.65 — still high after 1 quick epoch |
| **Val loss (`loss`)** | Same loss formula on **validation** images (not in mini-train subset) | 4.03 — reference for generalization |
| **`class_accuracy`** | % of val images where **product type** is exactly right | **76%** — best task so far |
| **`gender_accuracy`** | % where **gender** is right | **57%** |
| **`color_accuracy`** | % where **color** is right | **19%** — hardest here (many color labels) |
| **`*_macro_f1`** | Score that treats **each label equally** (good for imbalanced classes) | Class F1 **71%** > gender **44%** > color **12%** |
| **`joint_accuracy`** | % where **class AND gender AND color** are all right on the **same** photo | **6.8%** — strict; (0.76 × 0.57 × 0.19 ≈ 8%) ballpark |

**Accuracy vs macro F1:** If the model always predicts "Shirt," accuracy can look okay when Shirts are common; macro F1 punishes ignoring rare labels.

**Why joint is so low:** You must win multiple quizzes on one image. Even decent per-task scores multiply to a small number.

**After mini-training only:** Do not expect production quality. Use charts in notebook **§8b** (`plot_mini_training_metrics`) or full training + history plots (§10).

### Train loss / val loss — scale and how to judge 4.65 vs 4.03

**Not a percentage.** Loss is an open-ended penalty score (lower = better). There is no fixed "100% loss."

**What you add up** (`src/train.py` + `config.yaml`):

```text
total_loss = 1.0 × loss_class + 0.7 × loss_gender + 1.2 × loss_color + 0.6 × loss_usage
```

Each piece is **cross-entropy** for one head (how badly wrong the class scores are).

**Anchors:**

| Total loss (ballpark) | Meaning |
|-----------------------|---------|
| **~7+** | Near random / barely trained |
| **~3–5** | Learning; mini-run territory |
| **~1–2** | Decent after real training (depends on data) |
| **→ 0** | Perfect (never exactly 0 in practice) |

**How to evaluate 4.65 vs 4.03:**

1. **Direction over time** — loss should **drop** across epochs (main use in training).
2. **Train vs val** — val **4.03** < train **4.65** here → no overfitting on this tiny run; gap may grow if model memorizes.
3. **Use accuracy/F1 for "how good"** — loss is for the optimizer; humans read **76% class**, **7% joint**, etc.
4. **Compare to ~7.0 random baseline**, not to 0.

**Per-image intuition:** CE ≈ `-log(probability on true label)`. Confident and wrong → huge loss; confident and right → small loss.

### My notes — metrics

---

## Chapter 5c — Class imbalance (why it matters and how we fix it)

### The problem, visually

```text
Dataset distribution:
  Shirts:     ████████████████████████████████  8,000
  T-Shirts:   ████████████████████████          6,000
  Shoes:      ████████████████████              5,000
  Pants:      ████████████████                  4,000
  Watches:    ██████████                        2,500
  Bags:       ████████                          2,000
  Sunglasses: ████                              1,000
  Caps:       ██                                  500
```

Without correction, the model will:
- See 16x more shirt images than cap images during training
- Learn that "guess Shirt" is a safe strategy (right 27% of the time just by frequency)
- Completely ignore rare classes like Caps (small penalty for getting them wrong)

### Fix 1: WeightedRandomSampler (your code uses this)

Instead of going through images in order, rare classes are sampled more frequently:

```text
Without sampler (natural order):
  Batch 1: Shirt, Shirt, Pants, Shirt, Shoes, Shirt, T-Shirt, Shirt...
  (Caps almost never appears)

With WeightedRandomSampler:
  Batch 1: Shirt, Caps, Shoes, Sunglasses, Pants, Bags, Caps, T-Shirt...
  (every class appears roughly equally)
```

**How the weight is calculated:**

```python
# For each image, weight = 1 / (count of its class)
class_counts = [8000, 6000, 5000, 4000, 2500, 2000, 1000, 500]
sample_weights = [1/count for count in class_counts]
# Caps images get weight 1/500 = 0.002
# Shirt images get weight 1/8000 = 0.000125
# Caps is 16x more likely to be sampled → balances out
```

### Fix 2: Class-weighted Cross-Entropy (your code uses this)

Even with balanced sampling, the loss can still favor common classes. Class-weighted CE adds a per-class multiplier:

```text
Loss for misclassifying a Shirt:  small weight × CE = mild penalty
Loss for misclassifying a Cap:    large weight × CE = severe penalty
```

### Fix 3: Macro F1 as the evaluation metric

Accuracy hides imbalance problems. Macro F1 exposes them:

```text
Model that always predicts "Shirt":
  Accuracy:  8000/29000 = 27.6%  (looks bad but not terrible)
  Macro F1:  Only 1 out of 8 classes gets any F1 → ~12.5% (exposed!)

Model that performs equally on all classes:
  Accuracy:  85%
  Macro F1:  85% (consistent)
```

### Why all three fixes together?

| Fix | What it solves | What it misses |
|-----|---------------|----------------|
| WeightedRandomSampler | Training sees each class equally | Doesn't change loss penalty |
| Class-weighted CE | Mistakes on rare classes cost more | Doesn't change sampling frequency |
| Macro F1 evaluation | Catches models that ignore rare classes | Doesn't fix training — only measures |

Together, they attack imbalance from three angles: sampling, loss, and evaluation.

### My notes — class imbalance

---

## Chapter 6 — Transfer Learning (deep dive)

### What is transfer learning?

```text
TRAINING FROM SCRATCH:
  Random weights → train on YOUR data → mediocre model
  (needs millions of images and days of training)

TRANSFER LEARNING:
  ImageNet weights → fine-tune on YOUR data → strong model
  (needs thousands of images and hours of training)
```

**Analogy:** A chef who learned to cook Italian food can learn Japanese food much faster than someone who has never cooked. The basic skills (knife work, heat control, seasoning) transfer.

### What did ImageNet teach the backbone?

ImageNet has 14 million images across 1000 categories. The backbone learned:

```text
Layer 1-3:   Edges (horizontal, vertical, diagonal)
Layer 4-8:   Textures (stripes, dots, fur, fabric)
Layer 9-15:  Parts (eyes, wheels, sleeves, buckles)
Layer 16+:   Objects (dog, car, shoe, watch)
```

**The early/middle features are universal.** Edges and textures exist in fashion images too. That's why transfer learning works — you don't need to re-learn "what an edge is."

### Your training strategy: end-to-end fine-tuning

```text
Option A: Feature extraction (freeze backbone)
  Backbone weights: FROZEN (don't change)
  Only heads train
  Pro: Fast, works with small data
  Con: Backbone features might not fit fashion domain well

Option B: End-to-end fine-tuning (YOUR APPROACH)
  Backbone weights: TRAINABLE (adjust to fashion data)
  Heads also train
  Pro: Backbone adapts to fashion-specific features
  Con: Needs more data to avoid overfitting

Option C: Staged (freeze first, then unfreeze)
  Phase 1: Freeze backbone, train heads (2 epochs)
  Phase 2: Unfreeze last N layers, train everything (3 epochs)
  Pro: Stable head training first, then careful backbone adjustment
  Con: More complex, two learning rate schedules
  Status: PLANNED but not implemented in your code
```

**Your code uses Option B.** The `--no-pretrained` flag exists but is not the default. By default, `pretrained=True` loads ImageNet weights, and all parameters are trainable.

**Why Option B worked well enough:** With ~44K images in the dataset, there's enough data for end-to-end fine-tuning without severe overfitting. Early stopping (patience=3) prevents the model from memorizing. 5 epochs is short enough that the backbone adjusts gently rather than drastically.

### One-line answer for your report

> "Training strategy: Transfer learning from ImageNet-pretrained CNN backbones with end-to-end fine-tuning of backbone and multi-task heads on a filtered fashion catalog dataset, using weighted sampling and multi-task weighted cross-entropy."

### My notes — transfer learning

---

## Chapter 7 — Evaluation Metrics (full guide)

### The hierarchy of metrics

```text
BASIC:          Accuracy (% correct)
                    ↓ not enough because of class imbalance
BETTER:         Precision, Recall, F1 (per class)
                    ↓ need to aggregate across classes
STANDARD:       Macro F1 (average F1 across all classes equally)
                    ↓ need to evaluate multiple tasks together
PROJECT-SPECIFIC: Joint Accuracy (all 4 heads correct on same image)
```

### Precision and Recall — the two sides of a coin

**Precision:** "Of everything I predicted as Blue, how many were actually Blue?"

```text
I said "Blue" 100 times.
85 of those were actually Blue.
Precision = 85/100 = 85%

High precision = few false positives (rarely wrong when I say Blue)
```

**Recall:** "Of all actual Blue images, how many did I catch?"

```text
There are 120 actually-Blue images.
I found 85 of them.
Recall = 85/120 = 71%

High recall = few false negatives (rarely miss a Blue image)
```

**The tension:** You can have high precision by being very conservative (only predict Blue when 99% sure → misses many), or high recall by being liberal (predict Blue often → many false positives). F1 balances both.

### F1 Score

```text
F1 = 2 × (precision × recall) / (precision + recall)
F1 = 2 × (0.85 × 0.71) / (0.85 + 0.71) = 0.774
```

F1 is the harmonic mean — it punishes large gaps between precision and recall. If either is low, F1 is low.

### Macro F1 vs Micro F1 vs Weighted F1

```text
Per-class F1 scores:
  Black:  0.82     (3000 samples)
  Blue:   0.77     (2000 samples)
  White:  0.80     (1500 samples)
  Pink:   0.45     (200 samples)   ← rare class, model struggles

Macro F1:    (0.82 + 0.77 + 0.80 + 0.45) / 4 = 0.71
  → Treats every class equally regardless of size
  → Pink drags the score down (good — exposes weakness)

Micro F1:    Compute precision/recall on ALL samples pooled
  → Dominated by large classes (Black, Blue)
  → Pink barely affects the score (bad — hides weakness)

Weighted F1: Weight by class size
  → Middle ground
```

**Your project uses Macro F1** because you want to know if the model works for ALL categories, not just common ones.

### Joint Accuracy — the strictest metric

```text
Image 1: Class=Shirt ✓, Gender=Men ✓, Color=Blue ✓, Usage=Casual ✓  → Joint: ✓
Image 2: Class=Shoes ✓, Gender=Women ✓, Color=Red ✗, Usage=Sports ✓  → Joint: ✗
Image 3: Class=Watch ✓, Gender=Men ✗, Color=Gold ✓, Usage=Formal ✓  → Joint: ✗
```

Even with per-head accuracies of 98%, 87%, 65%, 86%, joint accuracy is:
0.98 × 0.87 × 0.65 × 0.86 ≈ **47.7%**

This is why your ~48% joint accuracy is actually **consistent** with the per-head numbers. It's not a sign that the model is bad — it's just a very strict metric.

**Why track it?** In catalog enrichment, a product entry with correct class but wrong color is still a bad entry. Joint accuracy tells you how often the full enrichment is correct.

### Confusion Matrix — where the model gets confused

```text
                    Predicted:
                    Black  Blue  White  Red   Grey
Actual: Black       [85]    3     5     1     6     ← 6 confused with Grey!
        Blue          2   [72]    4    10     2     ← 10 confused with Red!
        White         5     2   [80]    0     3
        Red           1    12     0   [65]    2     ← 12 confused with Blue!
        Grey          8     1     4     0   [67]    ← 8 confused with Black!
```

Common confusion pairs in fashion:
- Blue ↔ Navy (both blue-ish)
- Black ↔ Grey ↔ Dark Brown (all dark)
- White ↔ Beige ↔ Cream (all light)
- Red ↔ Pink ↔ Maroon (all reddish)

This is why your color accuracy is ~65% and not 95% — many colors are genuinely ambiguous, especially in compressed/pixelated product images.

### IoU and mAP — detection metrics (NOT used in your V1)

**IoU (Intersection over Union):** Measures how well a predicted bounding box overlaps the true box.

```text
IoU = (area where boxes overlap) / (area of both boxes combined)

IoU = 1.0 → perfect box placement
IoU = 0.5 → decent (common threshold for "correct")
IoU = 0.0 → boxes don't overlap at all
```

**mAP (Mean Average Precision):** Average detection precision across all classes and IoU thresholds. The standard benchmark for YOLO, Faster R-CNN, etc.

**Why not used in your project:** You classify whole images, not bounding boxes. IoU/mAP only apply if you add the YOLOv8 detection stage in V2.

### My notes — evaluation

---

## Chapter 8 — Inference

Load `best_model.pt` → same transforms as validation (no random flip) → print predicted class, gender, color, usage.

File: `src/inference.py`

### What happens during inference

```text
1. Load saved model weights (checkpoint)
2. Set model to eval mode (turns off dropout, uses running stats for batch norm)
3. For each image:
   a. Apply validation transforms (resize, normalize — NO augmentation)
   b. Convert to tensor [1, 3, 224, 224]
   c. Start TAT timer
   d. model(tensor) → four sets of logits
   e. Stop TAT timer
   f. Softmax on each logit → probabilities
   g. Pick highest probability per head → predicted label + confidence
4. Return JSON with labels, confidences, TAT
```

### torch.no_grad() — why it matters

During training: PyTorch tracks every operation to compute gradients later (backward pass). This uses memory and CPU time.

During inference: You don't need gradients (no learning happening). `torch.no_grad()` tells PyTorch to skip tracking, saving ~30-50% memory and speeding up the forward pass.

```python
with torch.no_grad():  # ← this single line speeds up inference
    outputs = model(image_tensor)
```

### model.eval() vs model.train()

```text
model.train():
  - Dropout is ACTIVE (randomly drops neurons — regularization)
  - BatchNorm uses batch statistics
  - Used during training

model.eval():
  - Dropout is DISABLED (all neurons active — consistent predictions)
  - BatchNorm uses running average statistics (stable)
  - Used during inference/evaluation
  - ALWAYS call this before predicting!
```

### Confidence scores — what they mean

The softmax output gives probabilities for each label:

```text
Color head softmax: [0.03, 0.85, 0.02, 0.05, 0.01, 0.01, 0.02, 0.01, 0.00, 0.00]
                            ↑
                          "Blue" = 85% confidence

Interpretation:
  85% → Model is quite confident this is Blue
  But 5% on index 3 (Red?) → some visual similarity
```

**High confidence (>80%):** Model is sure. Usually correct.
**Medium confidence (50-80%):** Model sees some ambiguity. Worth checking.
**Low confidence (<50%):** Model is uncertain. Likely a hard image or out-of-distribution. In your active learning loop, these get flagged for human review.

### My notes — inference

---

## Chapter 9 — ONNX and Optimization

### What is ONNX?

**ONNX** = Open Neural Network Exchange. A universal format for AI models.

```text
PyTorch model (.pt)  ──export──→  ONNX model (.onnx)  ──run with──→  ONNX Runtime
```

Think of it like saving a Word doc as PDF — the content is the same, but PDF opens faster in some viewers.

### Why ONNX Runtime is faster on CPU

PyTorch is designed for **training** (flexibility, gradient tracking). ONNX Runtime is designed for **inference** (speed):

| Optimization | What ONNX Runtime does |
|-------------|----------------------|
| Operator fusion | Combines Conv + BatchNorm + ReLU into one operation |
| Memory planning | Reuses memory buffers between layers |
| CPU-specific instructions | Uses AVX/SSE/NEON instructions on your specific CPU |
| Graph optimization | Removes unnecessary operations, reorders for efficiency |

### Your project's ONNX results

```text
MobileNetV3 on EC2 CPU:
  PyTorch:      16.5 ms/image   (flexible but slow on CPU)
  ONNX Runtime:  6.8 ms/image   (optimized for CPU) → 2.42x faster!

MobileNetV3 on Mac (MPS GPU):
  PyTorch:       0.7 ms/image   (GPU is blazing fast)
  ONNX Runtime:  4.4 ms/image   (falls back to CPU) → SLOWER

Lesson: ONNX wins on CPU, PyTorch+GPU wins on GPU
```

### Dynamic vs Static Quantization

**Dynamic quantization** (what your project uses):
- Converts weights from FP32 → INT8 at load time
- Activations computed in FP32, quantized on-the-fly
- Simpler to implement (one line of code)

**Static quantization:**
- Requires a calibration dataset
- Pre-computes activation ranges
- Potentially faster but more complex setup

### ONNX export with dynamic batch size

Your code exports with `dynamic_axes` so the batch dimension is flexible:

```python
torch.onnx.export(
    model, dummy_input,
    "model.onnx",
    dynamic_axes={"input": {0: "batch_size"},  # batch dim can vary
                  "output": {0: "batch_size"}}
)
```

This means the same ONNX model can process 1 image or 32 images at once.

### My notes — ONNX

---

## Chapter 10 — Production and Active Learning

### Why production is different from a demo

```text
DEMO (Streamlit):
  1 user → uploads 5 images → waits → sees results
  Sequential, synchronous, simple

PRODUCTION (1M images/day):
  Catalog team dumps 50,000 new images daily
  Each image needs Class + Gender + Color + Usage
  Results stored in database for the e-commerce site
  Low-confidence items routed to human reviewers
  Model retrains monthly with new reviewed data
```

### Production architecture

```text
S3 bucket (images)
    ↓
Message Queue (Kafka/SQS) — holds image jobs
    ↓
Worker Pool (GPU servers) — processes batches of 256
    ↓
Results Database — stores predictions
    ↓
┌─────────────────────────┐
│ High confidence (>80%)  │ → Auto-enrich catalog
│ Low confidence (<60%)   │ → Human review queue
└─────────────────────────┘
    ↓
Reviewed labels → add to training data → retrain model
```

### Active Learning — the self-improving loop

```text
Model predicts → Confident? → YES → Auto-accept
                     ↓
                    NO
                     ↓
              Human reviews & corrects
                     ↓
              Corrected labels stored
                     ↓
              Added to training data
                     ↓
              Model retrained (better!)
                     ↓
              Fewer uncertain predictions next time ←──┘
```

**Why it matters:** You don't label 44,000 images by hand for every retraining cycle. The model labels most of them. Humans only check the uncertain ones (~10-20%). Over time, that percentage drops.

### My notes — production

---

## Chapter 11 — Detection vs Classification

### When each approach fits

| Scenario | Best approach |
|----------|--------------|
| Catalog photo (one product, clean background) | Classification ✓ (your V1) |
| User-uploaded outfit photo (multiple items, messy background) | Detection → Classification |
| Social media image (products mixed with people, furniture) | Detection → Classification |

### Why V1 uses classification only

The Kaggle dataset has clean, centered product photos. Each image = one product. Detection would add:
- Extra model (YOLO) = more complexity
- Extra latency (~20-50ms per image for detection)
- Need for bounding box annotations (dataset doesn't have them)

For catalog-style images, classification directly gives you the answer with less overhead.

### V2 detection pipeline (planned)

```text
Messy photo
    ↓
YOLOv8 → finds products, outputs bounding boxes
    ↓
Crop each product region
    ↓
Multi-head classifier → Class/Gender/Color/Usage per product
    ↓
Combine results
```

### My notes — detection

---

## Chapter 12 — TAT, batch efficiency, quantization, pruning (and where they live in code)

> **Kid version:** TAT = stopwatch for one answer. Batch = school bus for many photos. Quantization = simpler numbers in the backpack. Pruning = cutting unused wires (we mostly use a *smaller bus* instead).

### How the four ideas connect

```text
  Upload / folder of images
           │
           ├─► [Batch efficiency]  train & eval use batch_size=32 on GPU
           │                       Streamlit uses batch inference for N uploads
           │
           ├─► [TAT measured]      time around model.forward in inference.py
           │
           ├─► [Quantization]      ONNX INT8 export (optimize_onnx.py)
           │                       EC2 deployment uses ONNX Runtime
           │
           └─► [Pruning]           NOT implemented in this repo yet
                                   we use smaller backbones (MobileNet) instead
```

---

### 1. TAT (Turnaround Time) — **implemented**

**Plain meaning:** Milliseconds from "start forward pass" to "have class + gender + color + usage."

**Kid analogy:** Time from pressing "Go" on the quiz until all four answers are on the paper.

| Where | What it does |
|-------|--------------|
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
  → model(tensor)  → four heads
  → stop timer → tat_ms
```

**Run from terminal:**

```bash
python -m src.inference --checkpoint outputs/models/best_model_efficientnet_b0.pt --image path/to.jpg
python -m src.benchmark_latency --checkpoint outputs/models/best_model_efficientnet_b0.pt --image_dir data/sample_images
```

**What TAT does *not* include today:** disk read, JPEG decode, or Streamlit UI paint — only the neural net forward pass (plus tensor prep right before it).

---

### 2. Batch efficiency — **implemented**

**Plain meaning:** Processing **many images in one GPU trip** is cheaper **per image** than one trip per image.

**Kid analogy:** One bus for 32 kids vs 32 buses for 32 kids.

#### Where we use real batches

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

### 3. Quantization — **implemented (ONNX dynamic quantization)**

**Plain meaning:** Store weights with **smaller integers** (e.g. INT8) instead of big floats → smaller file, often faster on CPU.

**Kid analogy:** Cheat sheet with rounded numbers instead of a huge textbook.

| Where | What |
|-------|------|
| `configs/config.yaml` → `optimization.quantize_dynamic: true` | Turn on/off INT8 step |
| `configs/config.yaml` → `optimization.onnx_opset: 17` | ONNX version for export |
| `src/optimize_onnx.py` | 1) Export PyTorch → `model.onnx` 2) `quantize_dynamic(..., QInt8)` → `model_quantized.onnx` |
| `src/model.py` → `ONNXExportWrapper` | Flattens four heads for ONNX export |
| `outputs/metrics/onnx_optimization_report.json` | File sizes + status |

**Run:**

```bash
python -m src.optimize_onnx \
  --checkpoint outputs/models/best_model_efficientnet_b0.pt \
  --output_dir outputs/models
```

**EC2 deployment uses ONNX Runtime** for serving, which is the production-relevant optimization. The deployed Streamlit app loads `.onnx` models on EC2 and `.pt` models on Mac.

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
|------|--------------|--------|
| **TAT** | `src/inference.py`, `app/streamlit_app.py`, `src/benchmark_latency.py` | `streamlit run app/streamlit_app.py` |
| **Batch (train/eval)** | `src/train.py`, `src/evaluate.py`, `configs/config.yaml` | `python -m src.train --backbone efficientnet_b0` |
| **Batch (inference)** | `src/inference.py` → `predict_batch_pil` | Upload N images in Streamlit |
| **Quantization** | `src/optimize_onnx.py`, `configs/config.yaml` | `python -m src.optimize_onnx` |
| **Pruning** | — | Not implemented |

### Compare backbones for speed vs accuracy

Same code path, different checkpoint size / speed:

| Checkpoint | Role |
|------------|------|
| `best_model_efficientnet_b0.pt` | Balanced accuracy |
| `best_model_mobilenet_v3_large.pt` | Best quality + fastest on EC2 → **production default** |
| `best_model_resnet50.pt` | Largest file, highest class accuracy, slowest inference |

```bash
python -m src.benchmark_latency --checkpoint outputs/models/best_model_mobilenet_v3_large.pt
python -m src.compare_models --skip-eval   # accuracy side-by-side
```

### My notes — speed & deployment

<!-- Add your measured TAT numbers, ONNX sizes, ideas for true batched inference -->

---

## Chapter 13 — Trade-offs Reference Table

Every design decision in the project involves a trade-off. This table collects them all:

| Decision | Option A | Option B | Your choice | Why |
|----------|----------|----------|-------------|-----|
| Training strategy | Feature extraction (freeze backbone) | End-to-end fine-tuning | End-to-end | More expressive; 44K images is enough data; 5 epochs prevents overfitting |
| Backbone | ResNet50 (big, proven) | MobileNetV3 (small, fast) | MobileNetV3 | Better color/joint accuracy + 2.42x faster on EC2; class accuracy still 98% |
| Inference engine | PyTorch everywhere | ONNX on server, PyTorch on Mac | Platform-dependent | ONNX faster on CPU; PyTorch+MPS faster on GPU. Use the best tool for each environment |
| Color loss weight | Equal (1.0) | Higher (1.2) | 1.2 | Color is hardest; extra gradient signal needed. Higher than 1.2 hurts class accuracy |
| Class imbalance | Just accuracy | WeightedSampler + weighted CE + macro F1 | All three | Each solves a different angle of the problem |
| Detection vs classification | YOLOv8 first | Direct classification | Classification | Clean catalog images don't need detection. Saves complexity + latency. YOLOv8 planned for V2 |
| Number of heads | Separate models per task | Single multi-head model | Multi-head | Shared features, 1 forward pass, lower latency, consistent predictions |
| Epochs | Many (20-50) | Few (5) | 5 with early stopping | Pretrained weights need few epochs; more risks overfitting on this dataset |
| Color classes | Keep all raw colors | Group to 10 + Other | Grouped | Rare colors destabilize training. "Other" catch-all is practical |
| Bonus attribute | None (only 3 heads) | Usage (4th head) | Usage | Already in metadata, clean labels, relevant for catalog, low risk |

### My notes — trade-offs

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
| Softmax | Converts logits to probabilities that sum to 1.0 |
| Cross-Entropy | Loss function for classification; = -log(probability of correct class) |
| Loss | How wrong the model is (lower = better) |
| Gradient / backward | Directions to improve weights |
| Optimizer | Applies small weight updates |
| Adam | Popular optimizer that adapts learning rate per weight |
| Learning rate | Step size for weight updates (too big = unstable, too small = slow) |
| Epoch | One lap through all training data |
| Early stopping | Stop training when validation score stops improving |
| Overfitting | Model memorizes training data but fails on new data |
| Backbone | Image feature extractor (MobileNetV3, ResNet50, etc.) |
| Head | Classifier for one task (Linear layer) |
| Embedding | Compact numerical summary of an image (output of backbone) |
| Pretrained | Already learned general vision from ImageNet |
| Fine-tuning | Adjusting pretrained weights on new data |
| Checkpoint | Saved model file (.pt) |
| Convolution | Sliding filter operation that detects features |
| Pooling | Shrinking operation (max/avg) that reduces spatial size |
| Global Average Pooling | Averages each feature map to one number → creates embedding |
| Skip connection | Shortcut that lets gradients flow past layers (ResNet) |
| Depthwise separable conv | Efficient convolution variant (MobileNet) — splits into 2 cheaper steps |
| Dropout | Randomly disable neurons during training (regularization) |
| Batch normalization | Normalize activations between layers for stable training |
| Precision | Of predicted positives, how many are correct? |
| Recall | Of actual positives, how many did we catch? |
| F1 score | Harmonic mean of precision and recall |
| Macro F1 | Average F1 across all classes equally (penalizes ignoring rare classes) |
| Joint accuracy | % where ALL heads are correct on the same image |
| Confusion matrix | Table showing predicted vs actual labels — reveals systematic errors |
| IoU | Intersection over Union — bounding box overlap metric (detection only) |
| mAP | Mean Average Precision — detection benchmark (detection only) |
| TAT | Turnaround time (ms) for one prediction forward pass |
| Batch efficiency | Lower ms per image when many images share one `model()` call |
| Quantization | Smaller numeric weights (e.g. INT8) for faster/smaller deploy |
| Pruning | Remove weak weights after training (not in this repo yet) |
| ONNX | Exported model format for deployment runtimes |
| ONNX Runtime | Microsoft's inference engine optimized for speed |
| MPS | Metal Performance Shaders — Apple's GPU acceleration for Mac |
| ImageNet | 14M image dataset used to pretrain backbones |
| WeightedRandomSampler | PyTorch tool that oversamples rare classes during training |
| Class-weighted CE | Cross-entropy where rare classes have higher penalty |
| Active learning | Loop where model flags uncertain predictions for human review → retrain |
| Drift detection | Monitoring for distribution changes in production data |
| P95 latency | 95th percentile — 95% of requests are faster than this value |
| Triton | NVIDIA's production inference server for GPU serving |

---

## Notebook section → file map

| Notebook § | Source files |
|------------|-------------|
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
| 2026-05-30 | Chapter 5a: Loss functions deep dive (CE, weighted CE, multi-task loss, task weights, two-level weighting) |
| 2026-05-30 | Chapter 5c: Class imbalance — full explanation of three-fix approach |
| 2026-05-30 | Chapter 6: Transfer learning — three training strategies, why end-to-end was chosen |
| 2026-05-30 | Chapter 3 expanded: convolution, pooling, global avg pooling, linear layers, softmax, backbone comparisons |
| 2026-05-30 | Chapter 7: Evaluation metrics — precision, recall, F1, macro F1, joint accuracy, confusion matrix, IoU/mAP |
| 2026-05-30 | Chapter 8: Inference — torch.no_grad, model.eval, confidence scores |
| 2026-05-30 | Chapter 9: ONNX — why it's faster, quantization types, dynamic batch |
| 2026-05-30 | Chapter 10: Production architecture and active learning loop |
| 2026-05-30 | Chapter 11: Detection vs classification — when each fits |
| 2026-05-30 | Chapter 13: Trade-offs reference table — every design decision documented |
| 2026-05-30 | Updated loss weights to actual values (1.0/0.7/1.2/0.6), added 4th head (usage) throughout |
| 2026-05-30 | Expanded glossary with ~50 terms |
| 2026-05-30 | Added data augmentation reasoning, normalization deep dive, label mapping examples |
