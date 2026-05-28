# Deployment plan — EC2 Streamlit (port 8789) + GitHub

> Plan only. No automation scripts committed yet.

---

## Part 1 — EC2: Streamlit prediction only

### What runs on EC2 (minimal)

You do **not** need training data, notebooks, or full `data/raw/` on the server.

| Required on EC2 | Why |
|-----------------|-----|
| `app/streamlit_app.py` | UI |
| `src/` (inference stack) | `inference.py`, `model.py`, `dataset.py` (transforms), `utils.py`, `tasks.py`, `active_learning.py`, `train_all.py` (backbone list only) |
| `configs/config.yaml` | Paths + inference settings |
| `outputs/models/best_model_*.pt` | **At least one** trained checkpoint (~16–95 MB each) |
| `requirements.txt` (or slim `requirements-inference.txt`) | Python deps |
| Optional: `outputs/metrics/evaluation_test_*.json` | Sidebar test scores in UI |

**Not needed for prediction-only:**

- `data/raw/`, `data/processed/` (labels live inside `.pt` checkpoint)
- `src/train.py`, `train_all`, `compare_models` (unless you train on EC2 later)
- Kaggle credentials

### Inference dependency chain

```text
streamlit_app.py
  → FashionCatalogPredictor (src/inference.py)
  → build_model_from_label_info (src/model.py)
  → build_transforms (src/dataset.py)
  → get_device (src/utils.py)   # CUDA on g4, else CPU on typical EC2
```

### EC2 instance sizing

| Instance | Inference | Notes |
|----------|-----------|--------|
| `t3.medium` (2 vCPU, 4 GB) | CPU, slow but OK for demo | Cheapest |
| `g4dn.xlarge` | CUDA GPU | Much faster TAT if you install CUDA PyTorch |
| Apple MPS | N/A on Linux EC2 | Your Mac training uses MPS; EC2 uses CUDA or CPU |

### Network: port 8789

1. **AWS Security Group** (EC2 → instance → Security):
   - Inbound: **TCP 8789** from your IP (recommended) or `0.0.0.0/0` (public demo — use only briefly).
2. **OS firewall** (if `ufw` enabled on Ubuntu):
   ```bash
   sudo ufw allow 8789/tcp
   ```
3. **Streamlit bind to all interfaces:**
   ```bash
   streamlit run app/streamlit_app.py \
     --server.port 8789 \
     --server.address 0.0.0.0 \
     --server.headless true
   ```
4. **Browser:** `http://<EC2_PUBLIC_IP>:8789`

Optional later: Nginx on 443 → proxy to 8789 + HTTPS (Let’s Encrypt).

### Step-by-step EC2 setup (Ubuntu 22.04 example)

#### A. One-time on EC2

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
cd ~
git clone <YOUR_GITHUB_REPO_URL> fashion_catalog_enrichment
cd fashion_catalog_enrichment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# CPU-only PyTorch (smaller install on non-GPU instances):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### B. Copy model weights (not in Git)

From your Mac:

```bash
scp -i ~/.ssh/your-key.pem \
  outputs/models/best_model_efficientnet_b0.pt \
  ubuntu@<EC2_IP>:~/fashion_catalog_enrichment/outputs/models/

# Optional: all three + metrics for sidebar
scp -i ~/.ssh/your-key.pem outputs/models/best_model_*.pt ubuntu@<EC2_IP>:~/fashion_catalog_enrichment/outputs/models/
scp -i ~/.ssh/your-key.pem outputs/metrics/evaluation_test_*.json ubuntu@<EC2_IP>:~/fashion_catalog_enrichment/outputs/metrics/
```

Or upload to **S3** and `aws s3 cp` on EC2 (better for large files / repeat deploys).

#### C. Run Streamlit (manual test)

```bash
cd ~/fashion_catalog_enrichment
source .venv/bin/activate
streamlit run app/streamlit_app.py --server.port 8789 --server.address 0.0.0.0 --server.headless true
```

#### D. Keep it running (systemd)

Create `/etc/systemd/system/fashion-streamlit.service`:

```ini
[Unit]
Description=Fashion Catalog Streamlit
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/fashion_catalog_enrichment
Environment=PATH=/home/ubuntu/fashion_catalog_enrichment/.venv/bin
ExecStart=/home/ubuntu/fashion_catalog_enrichment/.venv/bin/streamlit run app/streamlit_app.py --server.port=8789 --server.address=0.0.0.0 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable fashion-streamlit
sudo systemctl start fashion-streamlit
sudo systemctl status fashion-streamlit
```

### Security checklist (prediction demo)

| Item | Action |
|------|--------|
| SSH | Key-only, no password login |
| Port 8789 | Restrict source IP in security group if possible |
| Secrets | Never commit `kaggle.json`, `.env`, API keys |
| Upload limit | Streamlit already caps images in app (`max_upload_images`) |
| Auth | Streamlit has no built-in auth — add Nginx basic auth or Streamlit Cloud auth if public |

### Optional: slim `requirements-inference.txt`

For faster EC2 install, split deps (future commit):

```text
torch>=2.2.0
torchvision>=0.17.0
pandas numpy Pillow PyYAML streamlit
```

Drop: `matplotlib`, `onnx`, `onnxruntime`, `kaggle`, `scikit-learn` (not imported by inference path).

---

## Part 2 — Push to GitHub

### What Git should contain

| Include | Exclude (already in `.gitignore`) |
|---------|-----------------------------------|
| `app/`, `src/`, `configs/`, `scripts/`, `notebooks/*.md` | `.venv/`, `__pycache__/` |
| `requirements.txt`, `README.md`, `REPORT_TEMPLATE.md` | `data/raw/*`, `data/processed/*` |
| `.gitkeep` under empty dirs | `outputs/models/*.pt`, `*.onnx` |
| Sample docs / plans in `notebooks/` | `kaggle.json`, secrets |

**Model weights:** document in README that users must train locally or download artifacts from Releases / S3 (do not commit `.pt`).

### Pre-push checklist

1. **Secrets scan**
   ```bash
   git status
   # Ensure no kaggle.json, .env, API keys, large checkpoints staged
   ```
2. **Training finished** (optional): note in README which checkpoint is default (`efficientnet_b0`).
3. **Update README** with:
   - Clone + venv + train OR “download model from Releases”
   - EC2 pointer to this deploy doc
4. **Large files:** if you ever need a tiny sample model in repo, use **GitHub Releases** or **Git LFS** — not normal commits.

### GitHub workflow (first time)

```bash
cd /Users/atulagarwal/Documents/projects_scaler/fashion_catalog_enrichment_1

# Initialize if not already a repo
git init
git branch -M main

# Review what will be committed
git add app/ src/ configs/ scripts/ notebooks/ requirements.txt README.md REPORT_TEMPLATE.md .gitignore
git status   # double-check nothing sensitive

git commit -m "Fashion catalog enrichment: multi-head CV pipeline and Streamlit demo"

# Create empty repo on GitHub (web UI): fashion_catalog_enrichment — no README if you push local README

git remote add origin https://github.com/<YOUR_USER>/fashion_catalog_enrichment.git
git push -u origin main
```

**SSH alternative:**

```bash
git remote add origin git@github.com:<YOUR_USER>/fashion_catalog_enrichment.git
```

### Recommended repo hygiene

| Practice | Why |
|----------|-----|
| `main` = stable, working inference | EC2 clones `main` |
| Tag releases `v1.0.0` after training | Reproducible demos |
| GitHub Release attach `best_model_efficientnet_b0.pt` | Teammates / EC2 without retraining |
| `.github/workflows/` lint/test (later) | CI on push — optional |
| `DEPLOYMENT.md` or link to `notebooks/DEPLOYMENT_PLAN.md` | One place for EC2 steps |

### Connecting GitHub → EC2 (ongoing deploy)

```text
Mac: train → scp .pt to EC2 (or S3)
Mac: git push origin main
EC2: git pull && systemctl restart fashion-streamlit
```

Only restart Streamlit after code/config change; **new `.pt`** only needs `scp` + restart (or no restart if path unchanged).

---

## Suggested order of work

| Step | Task | Where |
|------|------|--------|
| 1 | Finish `train_all` + `compare_models` | Local Mac |
| 2 | Verify Streamlit locally | `streamlit run app/streamlit_app.py` |
| 3 | Create GitHub repo + first push (no `.pt`) | GitHub |
| 4 | Launch EC2, clone repo, venv, `pip install` | EC2 |
| 5 | `scp` checkpoints + optional metrics JSON | Mac → EC2 |
| 6 | Test `http://<IP>:8789` | Browser |
| 7 | systemd service for 24/7 | EC2 |
| 8 | (Optional) GitHub Release with model artifact | GitHub |

---

## Open choices

1. **Public repo vs private** — Private if course rubric allows; use Releases for model file sharing.
2. **One model on EC2** — Ship only `best_model_efficientnet_b0.pt` to save disk (~16 MB vs ~130 MB all three).
3. **HTTPS** — Required for production; for class demo, HTTP on 8789 is often enough.

---

## Revision log

| Date | Note |
|------|------|
| 2026-05-27 | Initial EC2 + GitHub deployment plan |
