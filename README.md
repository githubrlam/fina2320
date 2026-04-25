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

## 1. Download or Clone

If you are starting from GitHub, get the code first:

### Clone with Git

```bash
git clone https://github.com/githubrlam/fina2320.git
cd fina2320
```

### Or download as ZIP

1. Open the repository page: https://github.com/githubrlam/fina2320.git
2. Click the green Code button.
3. Choose Download ZIP.
4. Extract the ZIP file to a folder on your computer.
5. Open a terminal in the extracted repository root.

---

## 2. Prerequisites

- macOS, Linux, or Windows
- Python 3.10 or 3.11 recommended
- pip
- Optional but recommended: virtual environment support (`venv`)

Check your Python version:

```bash
python3 --version
```

---

## 3. Project Structure (Expected)

The script expects these CSV files:

- `nitendo.csv`
- `sony.csv`
- `capcom.csv`

Expected repository structure:

```text
repo-root/
  japan_gaming_pipeline_new.py
  nitendo.csv
  sony.csv
  capcom.csv
```

If your CSV files are somewhere else, set `DATA_DIR` accordingly (see Section 4).

---

## 4. Environment Setup

After cloning or downloading from GitHub, open a terminal at the repository root.

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

## 5. Configure Paths (Important)

Open `japan_gaming_pipeline_new.py` and check:

```python
DATA_DIR = ""
OUT_DIR  = ""
```

What these mean:

- `DATA_DIR`: folder prefix for input CSV files
- `OUT_DIR`: folder prefix for output figures

Because both are empty strings by default, the script reads/writes relative to your current working directory.

If you keep the defaults, run the script from the repository root so the CSV files are found automatically.

If you want to run from another folder, set the repository root explicitly:

```python
DATA_DIR = "/full/path/to/fina2320/"
OUT_DIR  = "/full/path/to/fina2320/"
```

Keep the trailing slash.

---

## 6. How To Run

### Option A (recommended with current defaults)

```bash
python japan_gaming_pipeline_new.py
```

### Option B (run from another folder after setting paths)

Use this only if you changed `DATA_DIR` and `OUT_DIR` to the correct repository path.

```bash
python /full/path/to/japan_gaming_pipeline_new.py
```

---

## 7. What The Script Produces

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

## 8. Common Issues

### Results are slightly different each run

You may see different metrics, probabilities, and final recommendations each time you run the pipeline. This is normal for deep-learning workflows.

Main reasons:

- Random weight initialization and mini-batch training dynamics
- Stochastic regularization (Dropout, SpatialDropout, GaussianNoise)
- Monte Carlo Dropout inference in Model 2 (`training=True` over multiple passes)
- Non-deterministic low-level math kernels (especially on GPU/parallel execution)

Notes:

- This script already sets TensorFlow and NumPy seeds, which improves consistency.
- Even with seeds, exact bit-for-bit reproducibility is not always guaranteed across different hardware/OS/library versions.

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

## 9. Quick Start (Copy/Paste)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib tensorflow tf-keras
python japan_gaming_pipeline_new.py
```

---

## 10. Optional: Freeze Dependencies

After successful setup:

```bash
pip freeze > requirements.txt
```

Then later recreate with:

```bash
pip install -r requirements.txt
```
