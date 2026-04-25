# Japan Gaming Stocks Neural Network Pipeline

This project trains and evaluates two deep-learning models on daily Japanese gaming stock data:

- Model 1: TCN regression for 21-day and 63-day forward log-returns
- Model 2: Micro-Transformer classifier for SELL / HOLD / BUY signals

Target stocks:

- Nintendo (7974)
- Sony (6758)
- Capcom (9697)

Main script:

- `japan_gaming_pipeline_new.py`

---

## 1. Prerequisites

- macOS, Linux, or Windows
- Python 3.10 or 3.11 recommended
- pip
- Optional but recommended: virtual environment support (`venv`)

Check your Python version:

```bash
python3 --version
```

---

## 2. Project Structure (Expected)

The script expects these CSV files:

- `nitendo.csv`
- `sony.csv`
- `capcom.csv`

Expected repository structure:

```text
repo-root/
  files/
    japan_gaming_pipeline_new.py
    nitendo.csv
    sony.csv
    capcom.csv
```

If your CSV files are somewhere else, set `DATA_DIR` accordingly (see Section 4).

---

## 3. Environment Setup

After downloading from GitHub, open a terminal at the repository root.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow tf-keras
```

Notes:

- `tf-keras` is imported in the script as `tf_keras`.
- If TensorFlow installation fails on your machine, install a specific version such as:

```bash
pip install "tensorflow==2.16.*" "tf-keras==2.16.*"
```

---

## 4. Configure Paths (Important)

Open `files/japan_gaming_pipeline_new.py` and check:

```python
DATA_DIR = ""
OUT_DIR  = ""
```

What these mean:

- `DATA_DIR`: folder prefix for input CSV files
- `OUT_DIR`: folder prefix for output figures

Because both are empty strings by default, the script reads/writes relative to your current working directory.

If you set paths manually, set it according to your repository structure and keep the trailing slash.

---

## 5. How To Run

### Option A (recommended with current defaults)

```bash
python "file_path/japan_gaming_pipeline_new.py"
```

### Option B (run from inside `file` folder)

Use this if your terminal is already inside the script folder.

```bash
python japan_gaming_pipeline_new.py
```

---

## 6. What The Script Produces

Console output includes:

- Data loading summary
- Training logs for both models
- Test metrics (MAE, R², directional accuracy, CV accuracy)
- Final forward predictions from latest date

Figures generated:

- `rpt_00_pipeline.png`
- `rpt_m1_01_price_history.png`
- `rpt_m1_02_predictions.png`
- `rpt_m1_03_residuals.png`
- `rpt_m1_04_loss.png`
- `rpt_m1_05_forecast.png`
- `rpt_m2_01_cv_accuracy.png`
- `rpt_m2_02_loss_curves.png`
- `rpt_m2_03_confusion.png`
- `rpt_m2_04_probs.png`
- `rpt_m2_05_dashboard.png`

These are saved under `OUT_DIR` (or current folder if `OUT_DIR = ""`).

---

## 7. Common Issues

### TensorFlow install errors

Try:

```bash
pip install --upgrade pip setuptools wheel
pip install "tensorflow==2.16.*" "tf-keras==2.16.*"
```

### File not found for CSV

- Ensure filenames exactly match: `nitendo.csv`, `sony.csv`, `capcom.csv`
- Ensure your working directory and `DATA_DIR` align

### Slow training

- CPU training can be slow; this is expected
- You can lower epochs in the config for quick tests:
  - `M1_EPOCHS`
  - `M2_EPOCHS`

---

## 8. Quick Start (Copy/Paste)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib tensorflow tf-keras
cd "file_path"  # file_path depends on your case
python "japan_gaming_pipeline_new.py"
```

---

## 9. Optional: Freeze Dependencies

After successful setup:

```bash
pip freeze > requirements.txt
```

Then later recreate with:

```bash
pip install -r requirements.txt
```
