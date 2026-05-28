from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.active_learning import create_review_queue, should_flag_for_review
from src.inference import FashionCatalogPredictor
from src.train_all import BACKBONES

st.set_page_config(page_title="Fashion Catalog Enrichment", page_icon="🛍️", layout="wide")

MODEL_DIR = REPO_ROOT / "outputs" / "models"
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"
SAMPLE_IMAGES_DIR = Path(__file__).resolve().parent / "sample_images"

BACKBONE_DISPLAY = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
    "mobilenet_v3_large": "MobileNet V3 Large",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def discover_checkpoints() -> list[tuple[str, str]]:
    """Return (sidebar label, checkpoint path relative to repo root)."""
    found: dict[str, Path] = {}
    for path in MODEL_DIR.glob("best_model_*.pt"):
        backbone = path.stem.removeprefix("best_model_")
        if backbone:
            found[backbone] = path

    options: list[tuple[str, str]] = []
    for backbone in BACKBONES:
        if backbone not in found:
            continue
        rel = found[backbone].relative_to(REPO_ROOT).as_posix()
        display = BACKBONE_DISPLAY.get(backbone, backbone)
        options.append((display, rel))

    default_pt = MODEL_DIR / "best_model.pt"
    if default_pt.exists() and not options:
        options.append(("Default", default_pt.relative_to(REPO_ROOT).as_posix()))
    return options


def discover_sample_images() -> list[Path]:
    """Return built-in demo images shipped with the app."""
    candidates: list[Path] = []

    # Preferred location: app/sample_images/*
    if SAMPLE_IMAGES_DIR.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            candidates.extend(sorted(SAMPLE_IMAGES_DIR.glob(ext)))

    # Fallback: any images directly under app/ (or nested one level)
    app_dir = Path(__file__).resolve().parent
    if not candidates:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            candidates.extend(sorted(app_dir.glob(ext)))
            candidates.extend(sorted(app_dir.glob(f"*/*{ext.lstrip('*')}")))

    # De-duplicate while preserving order
    seen: set[Path] = set()
    images: list[Path] = []
    for p in candidates:
        if p not in seen:
            images.append(p)
            seen.add(p)
    return images


@st.cache_data
def load_test_metrics(backbone: str) -> dict | None:
    metrics_path = METRICS_DIR / f"evaluation_test_{backbone}.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_predictor(checkpoint_path: str) -> FashionCatalogPredictor:
    return FashionCatalogPredictor(str(REPO_ROOT / checkpoint_path))


def confidence_badge(score: float, threshold: float) -> str:
    """Return a coloured emoji badge based on confidence level."""
    if score >= threshold:
        return "🟢"
    if score >= 0.50:
        return "🟡"
    return "🔴"


def confidence_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def styled_confidence(score: float, threshold: float) -> str:
    badge = confidence_badge(score, threshold)
    return f"{badge} {score:.0%}"


def prediction_to_row(result: dict, threshold: float) -> dict:
    p = result["prediction"]
    return {
        "Image": result["image_name"],
        "Class": p["class"]["label"],
        "Class Conf.": styled_confidence(p["class"]["confidence"], threshold),
        "Gender": p["gender"]["label"],
        "Gender Conf.": styled_confidence(p["gender"]["confidence"], threshold),
        "Color": p["color"]["label"],
        "Color Conf.": styled_confidence(p["color"]["confidence"], threshold),
        **(
            {
                "Usage": p["usage"]["label"],
                "Usage Conf.": styled_confidence(p["usage"]["confidence"], threshold),
            }
            if "usage" in p
            else {}
        ),
        "Overall": styled_confidence(result["overall_confidence"], threshold),
        "TAT ms": f"{result['tat_ms']:.1f}",
    }


# ── sidebar ───────────────────────────────────────────────────────────────────

checkpoint_options = discover_checkpoints()
if not checkpoint_options:
    st.sidebar.error("No trained models found in `outputs/models/`. Run training first.")
    st.stop()

labels = [label for label, _ in checkpoint_options]
paths = [path for _, path in checkpoint_options]
backbones = [Path(path).stem.removeprefix("best_model_") for path in paths]

default_index = 0
if "resnet50" in backbones:
    default_index = backbones.index("resnet50")
elif "efficientnet_b0" in backbones:
    default_index = backbones.index("efficientnet_b0")

with st.sidebar:
    st.header("⚙️ Settings")
    selected_label = st.selectbox(
        "Model",
        options=labels,
        index=default_index,
        help="Trained checkpoints from outputs/models/",
    )
    checkpoint_path = paths[labels.index(selected_label)]
    selected_backbone = backbones[labels.index(selected_label)]
    st.caption(f"`{checkpoint_path}`")

    max_images = st.number_input("Max images", min_value=1, max_value=10, value=10)
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.99, value=0.70, step=0.05,
        help="Below this → 🔴 low confidence and flagged for review"
    )
    show_top3 = st.toggle("Show top-3 alternatives per task", value=False)
    st.divider()
    st.markdown("**Test set scores**")
    test_metrics = load_test_metrics(selected_backbone)
    if test_metrics:
        st.markdown(f"Class accuracy **{test_metrics['class_accuracy']:.1%}**")
        st.markdown(f"Gender accuracy **{test_metrics['gender_accuracy']:.1%}**")
        st.markdown(f"Color accuracy **{test_metrics['color_accuracy']:.1%}**")
        if "usage_accuracy" in test_metrics:
            st.markdown(f"Usage accuracy **{test_metrics['usage_accuracy']:.1%}**")
        st.markdown(f"Joint accuracy (all heads) **{test_metrics['joint_accuracy']:.1%}**")
    else:
        st.caption("No evaluation_test_<model>.json — run `python -m src.compare_models`")

# ── main ──────────────────────────────────────────────────────────────────────

st.title("🛍️ Fashion Catalog Enrichment")
st.caption(
    f"Model: **{selected_label}** · Upload up to 10 product images → Class · Gender · Color · Usage · confidence · TAT"
)

if "results" not in st.session_state:
    st.session_state["results"] = None
if "last_checkpoint_path" not in st.session_state:
    st.session_state["last_checkpoint_path"] = None

sample_images = discover_sample_images()
input_mode = st.radio(
    "Choose input source",
    options=["Built-in sample images", "Upload my images"],
    horizontal=True,
)

uploaded_files = []
selected_sample_paths: list[Path] = []

if input_mode == "Upload my images":
    uploaded_files = st.file_uploader(
        "Upload product images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    ) or []
else:
    if not sample_images:
        st.warning(
            f"No sample images found in `{SAMPLE_IMAGES_DIR}`. "
            "Create that folder and add a few .jpg/.png/.webp files."
        )
    else:
        options = [p.name for p in sample_images]
        selected_names = st.multiselect(
            "Select sample images",
            options=options,
            default=options[: min(3, len(options))],
        )
        selected_sample_paths = [p for p in sample_images if p.name in set(selected_names)]

num_selected = len(uploaded_files) if input_mode == "Upload my images" else len(selected_sample_paths)
if num_selected > max_images:
    st.error(f"Please select up to {max_images} images only (got {num_selected}).")
    st.stop()

if input_mode == "Upload my images" and uploaded_files:
    st.caption(f"Selected: **{len(uploaded_files)}** upload(s)")
elif input_mode == "Built-in sample images" and selected_sample_paths:
    st.caption(f"Selected: **{len(selected_sample_paths)}** sample image(s)")
else:
    st.caption("Select images above, then click **Run prediction**.")

if not (REPO_ROOT / checkpoint_path).exists():
    st.error(f"Checkpoint not found: `{checkpoint_path}`. Train a model first or pick another model in the sidebar.")
    st.stop()

if st.session_state["last_checkpoint_path"] != checkpoint_path:
    st.session_state["results"] = None
    st.session_state["last_checkpoint_path"] = checkpoint_path

run_clicked = st.button("Run prediction", type="primary")
if run_clicked:
    if input_mode == "Upload my images":
        if not uploaded_files:
            st.error("Please upload at least one image before running prediction.")
            st.stop()
        images = [Image.open(file).convert("RGB") for file in uploaded_files]
        names = [file.name for file in uploaded_files]
    else:
        if not selected_sample_paths:
            st.error("Please select at least one sample image before running prediction.")
            st.stop()
        images = [Image.open(p).convert("RGB") for p in selected_sample_paths]
        names = [p.name for p in selected_sample_paths]

    predictor = load_predictor(checkpoint_path)
    batch_results = predictor.predict_batch_pil(images, names)
    st.session_state["results"] = {"images": images, "batch_results": batch_results}

state = st.session_state.get("results")
if not state:
    st.info("Click **Run prediction** to generate predictions.")
    st.stop()
    # In bare/non-Streamlit execution contexts, st.stop() may not halt execution.
    state = {"images": [], "batch_results": []}

images = state["images"]
batch_results = state["batch_results"]
results = list(zip(images, batch_results))

# ── summary banner ────────────────────────────────────────────────────────────

flagged = [r for _, r in results if should_flag_for_review(r, confidence_threshold)]
high_conf = [r for _, r in results if r["overall_confidence"] >= confidence_threshold]

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Images processed", len(results))
col_b.metric("High confidence 🟢", len(high_conf))
col_c.metric("Flagged for review 🔴", len(flagged))
avg_tat = batch_results[0]["tat_ms"] if batch_results else 0.0
batch_forward_ms = batch_results[0].get("batch_tat_ms", 0.0) if batch_results else 0.0
col_d.metric(
    "Avg TAT / image",
    f"{avg_tat:.1f} ms",
    help=f"One forward pass for {len(results)} image(s): {batch_forward_ms:.1f} ms total",
)

st.divider()

# ── card grid: image + predictions side by side ───────────────────────────────

st.subheader("📋 Results per image")

for image, result in results:
    p = result["prediction"]
    overall = result["overall_confidence"]
    badge = confidence_badge(overall, confidence_threshold)
    flag_text = " ⚠️ Flagged for review" if should_flag_for_review(result, confidence_threshold) else ""

    with st.expander(f"{badge} **{result['image_name']}** — overall confidence {overall:.0%}{flag_text}", expanded=True):
        img_col, pred_col = st.columns([1, 2])

        with img_col:
            st.image(image, use_container_width=True)
            st.caption(f"TAT (amortized): **{result['tat_ms']:.1f} ms** · batch forward **{result.get('batch_tat_ms', 0):.1f} ms**")

        with pred_col:
            for task in p:
                label = p[task]["label"]
                conf = p[task]["confidence"]
                b = confidence_badge(conf, confidence_threshold)
                st.markdown(
                    f"**{task.capitalize()}:** {label} &nbsp; {b} `{conf:.0%}`"
                )
                st.progress(min(conf, 1.0))

                if show_top3:
                    alts = p[task].get("top_predictions", [])
                    if len(alts) > 1:
                        alt_text = " · ".join(
                            f"{a['label']} ({a['confidence']:.0%})" for a in alts[1:]
                        )
                        st.caption(f"Alternatives: {alt_text}")

st.divider()

# ── flat summary table ────────────────────────────────────────────────────────

st.subheader("📊 Summary table")
rows = [prediction_to_row(r, confidence_threshold) for _, r in results]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── downloads ─────────────────────────────────────────────────────────────────

st.subheader("⬇️ Downloads")
dl_col1, dl_col2 = st.columns(2)

json_payload = json.dumps([r for _, r in results], indent=2)
dl_col1.download_button(
    "Download predictions JSON",
    data=json_payload,
    file_name="fashion_catalog_predictions.json",
    mime="application/json",
)

if flagged:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        review_rows = create_review_queue([r for _, r in results], tmp.name, confidence_threshold)
    dl_col2.download_button(
        f"Download review queue CSV ({len(flagged)} images)",
        data=pd.DataFrame(review_rows).to_csv(index=False),
        file_name="review_queue.csv",
        mime="text/csv",
    )
else:
    dl_col2.success("All images above confidence threshold — no review needed ✅")

# ── raw JSON (collapsible) ────────────────────────────────────────────────────

with st.expander("Raw JSON output"):
    st.code(json_payload, language="json")
