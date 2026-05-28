# Notebooks

Use this folder for exploratory and learning notebooks.

| Doc | Purpose |
|-----|---------|
| [ML_BASICS_REVISION.md](./ML_BASICS_REVISION.md) | ML/CNN revision notes tied to the walkthrough |
| [PROJECT_CHANGELOG.md](./PROJECT_CHANGELOG.md) | Timeline of repo changes and why they were made |
| [PLANNED_FEATURES.md](./PLANNED_FEATURES.md) | Design plan: usage head + dominant-color fallback |
| [DEPLOYMENT_PLAN.md](./DEPLOYMENT_PLAN.md) | EC2 Streamlit (port 8789) + GitHub push plan |

| Notebook | Purpose |
|----------|---------|
| `02_training_walkthrough.ipynb` | Step-by-step training pipeline (config → data → model → train). **Start here** to understand the codebase. |
| `01_eda.ipynb` | (Optional) EDA: class/gender/color distributions, sample images, split checks |

## Jupyter environment (one venv for terminal + kernel)

Use a single virtual environment at the **repo root** (`.venv`) so the integrated terminal, CLI scripts, and notebook kernel all share the same packages and Python version.

### 1. Create and activate the venv

From the repo root:

```bash
cd /path/to/fashion_catalog_enrichment_1

python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)
```

Confirm you are on the venv Python:

```bash
which python    # should end with .../fashion_catalog_enrichment_1/.venv/bin/python
python --version
```

### 2. Install project + Jupyter dependencies

Still with the venv activated:

```bash
pip install -U pip
pip install -r requirements.txt
pip install jupyterlab ipykernel
```

`ipykernel` registers this interpreter as a Jupyter kernel; `jupyterlab` (or `notebook`) is only needed if you run notebooks in the browser.

### 3. Register a named kernel (recommended)

This makes the kernel easy to pick in Cursor, VS Code, or Jupyter Lab:

```bash
python -m ipykernel install --user \
  --name fashion-catalog \
  --display-name "Python (fashion-catalog .venv)"
```

List installed kernels:

```bash
jupyter kernelspec list
```

You should see `fashion-catalog` pointing at your `.venv` Python.

To remove this kernel later:

```bash
jupyter kernelspec uninstall fashion-catalog
```

### 4. Use the same venv in Cursor / VS Code

**Integrated terminal**

1. Open a terminal in the editor (`Terminal` → `New Terminal`).
2. Run `source .venv/bin/activate` (once per terminal session), or set the workspace default:
   - Command Palette → **Python: Select Interpreter**
   - Choose `./.venv/bin/python`
   - New terminals will often auto-activate that venv when the Python extension is enabled.

**Notebook kernel**

1. Open a `.ipynb` file under `notebooks/`.
2. Click the kernel name in the top-right (or **Notebook: Select Notebook Kernel** from the Command Palette).
3. Pick either:
   - **Python (fashion-catalog .venv)** — the kernelspec from step 3, or
   - **`.venv` / `Python 3.x.x ('.venv': venv)`** — the workspace interpreter (same environment if step 4’s interpreter is `.venv/bin/python`).

If the kernel list is empty or wrong, run step 3 again with the venv activated, then reload the window (**Developer: Reload Window**).

**Imports from `src/`**

Run notebooks with the repo root as the working directory (default when you open the repo folder). The training walkthrough expects `from src...` to resolve; if imports fail, set the notebook folder’s parent as cwd or add the repo root to `PYTHONPATH`:

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

### 5. Run notebooks

**In the editor:** open `notebooks/02_training_walkthrough.ipynb`, select the kernel above, run cells.

**In the browser** (from repo root, venv activated):

```bash
jupyter lab notebooks/02_training_walkthrough.ipynb
# or: jupyter notebook notebooks/02_training_walkthrough.ipynb
```

Prerequisites: data preparation has been run at least once (see main [README](../README.md)).

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `bad interpreter: .../fashion_catalog_enrichment/.venv/...` | The `.venv` was copied or renamed from another folder. **Delete `.venv` and recreate** (steps 1–2). Do not copy a venv between projects. |
| Kernel not in the list | Activate `.venv`, re-run step 3, reload the editor window |
| Wrong packages in notebook | Kernel must be `.venv`; check `import sys; print(sys.executable)` in a cell |
| `ModuleNotFoundError: src` | Open the repo root as the workspace; run commands from repo root |
| `FileNotFoundError: data/raw/images/...` | Re-run notebook **setup cell** (it `chdir`s to repo root), or restart kernel after pulling latest `src/dataset.py` |
| Stale kernel after recreating `.venv` | Re-run steps 2–3; uninstall old kernelspec if the path changed |

## Run the training walkthrough (quick)

From the **repo root** with `.venv` activated:

```bash
source .venv/bin/activate
jupyter lab notebooks/02_training_walkthrough.ipynb
```

Or use the in-editor kernel steps in section 4–5 above.
