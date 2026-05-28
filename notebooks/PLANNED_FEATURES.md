# Planned features — Usage head

> **Decisions (2026-05-27):**
> - **Usage head:** Implement (4 labels + `Other` for rare/null) — **done in code**
> - **Dominant-color fallback:** **Not implementing** (cancelled)
> - **Joint accuracy:** class + gender + color + **usage** (all must match)

---

## Usage attribute — implemented

### Labels

| Model label | Raw `usage` values |
|-------------|-------------------|
| Casual | Casual, Smart Casual, Party, Travel |
| Sports | Sports |
| Formal | Formal |
| Ethnic | Ethnic |
| Other | Home, null, unmapped |

### Pipeline

```text
styles.csv (usage column)
  → data_preparation (usage_label, usage_idx)
  → 4th head in MultiHeadFashionClassifier
  → train / eval / inference / Streamlit
```

### Required before train

```bash
.venv/bin/python -m src.data_preparation --config configs/config.yaml
.venv/bin/python -m src.train_all --config configs/config.yaml --set-default efficientnet_b0
.venv/bin/python -m src.compare_models --split test
```

Old checkpoints (3-head only) still load in Streamlit but **without** usage predictions until retrained.

---

## Dominant-color fallback — cancelled

Not building per product decision. Color improvements rely on:

- Higher `loss_weights.color` (1.2)
- Retraining
- Active learning review queue on low confidence (existing)

---

## Revision log

| Date | Note |
|------|------|
| 2026-05-27 | Initial plan |
| 2026-05-27 | Usage implemented; color fallback cancelled |
