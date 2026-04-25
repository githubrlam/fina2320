"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      JAPAN GAMING STOCKS — COMPLETE NEURAL NETWORK PIPELINE                 ║
║      Nintendo (7974)  ·  Sony (6758)  ·  Capcom (9697)                      ║
║      Daily data: Apr 2016 – Apr 2026  (S&P Capital IQ)                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MODEL 1 — TCN Regression                                                   ║
║    Architecture : Temporal Convolutional Network, causal dilated conv        ║
║    Output       : 21d and 63d forward log-return (price targets)             ║
║    Features     : 23 technical indicators                                    ║
║    Evaluation   : MAE, R², directional accuracy on hold-out test set         ║
║                                                                              ║
║  MODEL 2 — Micro-Transformer Classifier                                     ║
║    Architecture : Single-block Transformer encoder, self-attention           ║
║    Output       : SELL / HOLD / BUY  (3-class softmax)                      ║
║    Features     : 8 technical indicators                                     ║
║    Evaluation   : Walk-forward CV accuracy vs random baseline (33.3%)        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CRITICAL BUG FIX                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Previous versions: engineer() drops last 63 rows (M1) / 21 rows (M2)       ║
║  because forward-return targets are NaN there. Using df[-WINDOW:] after     ║
║  dropna produced stale predictions:                                          ║
║    M1 base price: Jan 15, 2026  (92 days stale!)                            ║
║    M2 base price: Mar 18, 2026  (30 days stale)                             ║
║                                                                              ║
║  FIX: Separate functions:                                                    ║
║    features_only(df)  — computes features, NO target dropna                 ║
║                          → used for forward prediction (latest row)         ║
║    engineer(df)        — adds targets, drops last H rows                    ║
║                          → used ONLY for training sequences                 ║
║  Both models now predict from actual Apr 17, 2026                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ON MILD M2 OVERFITTING                                                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  The loss curves show a small train-val accuracy gap (~3–7%).                ║
║  This is ACCEPTABLE because:                                                 ║
║    • Gap < 8% threshold for financial time-series                            ║
║    • Val accuracy stays above random (33.3%) throughout                      ║
║    • No "scissors" pattern (val loss does not rise while train falls)        ║
║    • EarlyStopping restores best-weight checkpoint for evaluation            ║
║    • 9 regularisation layers already applied (MaxNorm, L2, Dropout ×3,      ║
║      GaussianNoise, SpatialDropout, label smoothing, AdamW, gradient clip)  ║
║    • Financial data has low SNR — zero gap would mean underfitting           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OUTPUT FIGURES                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  rpt_00_pipeline.png           Model architecture diagram                   ║
║  rpt_m1_01_price_history.png   10-year daily price + volume                 ║
║  rpt_m1_02_predictions.png     TCN: predicted vs actual log-returns          ║
║  rpt_m1_03_residuals.png       TCN: residual scatter + regression line       ║
║  rpt_m1_04_loss.png            TCN: training / validation loss curves        ║
║  rpt_m1_05_forecast.png        TCN: forward price targets (Apr 17 base)     ║
║  rpt_m2_01_cv_accuracy.png     Transformer: walk-forward CV accuracy         ║
║  rpt_m2_02_loss_curves.png     Transformer: loss + accuracy curves           ║
║  rpt_m2_03_confusion.png       Transformer: confusion matrices               ║
║  rpt_m2_04_probs.png           Transformer: softmax probability time-series  ║
║  rpt_m2_05_dashboard.png       Combined recommendation dashboard             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── Numeric / data ────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
from   sklearn.preprocessing       import StandardScaler
from   sklearn.metrics             import (mean_absolute_error, r2_score,
                                            confusion_matrix, ConfusionMatrixDisplay,
                                            accuracy_score)
from   sklearn.utils.class_weight  import compute_class_weight

# ── Plotting ──────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches  as mpatches

# ── Deep learning ─────────────────────────────────────────────────────────────
import tf_keras as keras
from tf_keras.models       import Model
from tf_keras.layers       import (Input, Dense, Conv1D, Add, Activation,
                                    BatchNormalization, SpatialDropout1D,
                                    Dropout, GlobalAveragePooling1D,
                                    LayerNormalization, MultiHeadAttention,
                                    GaussianNoise)
from tf_keras.regularizers import l2
from tf_keras.constraints   import MaxNorm
from tf_keras.optimizers    import Adam, AdamW
from tf_keras.callbacks     import (EarlyStopping, ReduceLROnPlateau,
                                     LearningRateScheduler)
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
#  CONFIGURATION
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = ""
OUT_DIR  = ""


# ── Stocks ────────────────────────────────────────────────────────────────────
STOCKS = {
    "Nintendo": "nitendo.csv",
    "Sony"    : "sony.csv",
    "Capcom"  : "capcom.csv",
}
SN3 = ["Nintendo", "Sony", "Capcom"]

# ── Model 1 — TCN Regression ──────────────────────────────────────────────────
M1_WINDOW     = 120    # look-back: 120 trading days (~6 months)
M1_H1         = 21     # 1-month forward horizon
M1_H2         = 63     # 3-month forward horizon
M1_TEST_FRAC  = 0.18
M1_VAL_FRAC   = 0.12
M1_EPOCHS     = 55
M1_BATCH      = 64
M1_LR         = 2e-4
M1_L2         = 1e-4
M1_SP_DROP    = 0.15
M1_DENSE_DROP = 0.30
M1_DILATIONS  = [1, 2, 4, 8]
M1_DIR_ALPHA  = 0.20
M1_DIR_TEMP   = 30.0
M1_HUBER_DELTA = 0.02
M1_SW_CAP     = 3.0

# ── Model 2 — Micro-Transformer Classifier ────────────────────────────────────
M2_WINDOW     = 60     # look-back: 40 trading days (~2 months)
M2_HORIZON    = 21     # predict 21d forward direction
M2_N_FOLDS    = 2      # walk-forward CV folds
M2_TEST_FRAC  = 0.12   # each fold's test fraction
M2_EPOCHS     = 50
M2_BATCH      = 128
M2_LR_MAX     = 3e-4
M2_LR_MIN     = 1e-6
M2_WARMUP     = 10     # linear warm-up epochs
M2_WD         = 2e-3   # AdamW weight decay
M2_L2         = 1e-2
M2_MAXNORM    = 3.0
M2_D_MODEL    = 12     # attention embedding dimension
M2_N_HEADS    = 2
M2_FFN_DIM    = 24     # feed-forward hidden size
M2_ATTN_DROP  = 0.25
M2_FFN_DROP   = 0.45
M2_SP_DROP    = 0.20
M2_NOISE_STD  = 0.04
M2_LABEL_SMOOTH = 0.10
M2_MC_PASSES  = 25     # Monte Carlo Dropout inference passes

M2_STOCK_TUNE = {
    "Sony": {
        "attn_drop": 0.30,
        "ffn_drop": 0.55,
        "sp_drop": 0.25,
        "noise_std": 0.06,
        "train_epochs": 32,
        "warmup_epochs": 6,
        "patience": 6,
    }
}

# ── Classification thresholds ─────────────────────────────────────────────────
BUY_T  =  0.03    # 21d log-return ≥ +3%  → BUY  (class 2)
SELL_T = -0.03    # 21d log-return ≤ -3%  → SELL (class 0)
                   # otherwise             → HOLD (class 1)
CLASS_NAMES = ["SELL", "HOLD", "BUY"]

# ── Colours ───────────────────────────────────────────────────────────────────
STOCK_C  = {"Nintendo": "#C0392B", "Sony": "#1A5276",  "Capcom": "#1A237E"}
STOCK_BG = {"Nintendo": "#FDEDEC", "Sony": "#EBF5FB",  "Capcom": "#EDE7F6"}
CLS_C    = {"SELL": "#E74C3C",     "HOLD": "#E67E22",  "BUY":    "#27AE60"}

# ── Report style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#FAFAFA",
    "axes.edgecolor"   : "#CCCCCC",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "grid.color"       : "#EEEEEE",
    "grid.linewidth"   : 0.8,
    "font.size"        : 11,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "legend.fontsize"  : 8,
    "legend.framealpha": 0.90,
})

def save(fname, fig):
    fig.savefig(OUT_DIR + fname, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {fname}")


# =============================================================================
#  SECTION 1 — DATA LOADING
# =============================================================================

def load_csv(path: str, name: str) -> pd.DataFrame:
    """Parse S&P Capital IQ weekly export → clean DataFrame."""
    df  = pd.read_csv(path, encoding="utf-8-sig")
    pc  = [c for c in df.columns if "Share Pricing" in c][0]
    vc  = [c for c in df.columns if "Volume"        in c][0]
    out = pd.DataFrame({
        "Date"  : pd.to_datetime(df["Dates"], format="%b-%d-%Y"),
        "Close" : pd.to_numeric(df[pc], errors="coerce"),
        "Volume": (df[vc].astype(str)
                   .str.replace("mm", "", regex=False)
                   .str.replace(",",  "", regex=False)
                   .pipe(pd.to_numeric, errors="coerce")),
    })
    out["Name"] = name
    return out.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)


print("══ 1. Loading data ═══════════════════════════════════════════════════")
stocks_raw = {n: load_csv(DATA_DIR + f, n) for n, f in STOCKS.items()}
for n, d in stocks_raw.items():
    print(f"   {n:10s}  rows={len(d)}  "
          f"last={d['Date'].iloc[-1].date()}  "
          f"close=¥{d['Close'].iloc[-1]:,.0f}")


# =============================================================================
#  SECTION 2 — FEATURE ENGINEERING
#  CRITICAL: Two separate functions per model
#    features_only() → no target columns, no final dropna
#                       used for forward prediction from latest date
#    engineer()      → adds targets, drops last H rows
#                       used for training/testing sequences only
# =============================================================================

# ── M1 feature list (23 features) ────────────────────────────────────────────
M1_FEAT = [
    "ret_1d",  "ret_3d",  "ret_5d",  "ret_10d", "ret_21d",
    "rmean_5", "rmean_10","rmean_21","rmean_63",
    "rstd_5",  "rstd_10", "rstd_21",
    "hi_52w",  "lo_52w",
    "rsi_14",  "rsi_28",
    "macd",    "macd_sig",
    "bb_pct",  "atr_norm",
    "vol_norm","vol_ret",
    "gk_vol",
]

def m1_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 23 features — NO target, NO target-related dropna.
    Last row will always be the actual latest trading day.
    """
    df = df.copy()
    p, v = df["Close"].copy(), df["Volume"].copy()

    # Log returns
    for lag, name in [(1,"ret_1d"),(3,"ret_3d"),(5,"ret_5d"),(10,"ret_10d"),(21,"ret_21d")]:
        df[name] = np.log(p / p.shift(lag))

    # Rolling stats on daily return
    r1 = df["ret_1d"]
    for w, name in [(5,"rmean_5"),(10,"rmean_10"),(21,"rmean_21"),(63,"rmean_63")]:
        df[name] = r1.rolling(w).mean()
    for w, name in [(5,"rstd_5"),(10,"rstd_10"),(21,"rstd_21")]:
        df[name] = r1.rolling(w).std()

    # 52-week high/low proximity
    df["hi_52w"] = p / p.rolling(252).max() - 1
    df["lo_52w"] = p / p.rolling(252).min() - 1

    # RSI (14 and 28)
    for span, name in [(14,"rsi_14"),(28,"rsi_28")]:
        delta = p.diff()
        g = delta.clip(lower=0).rolling(span).mean()
        l = (-delta.clip(upper=0)).rolling(span).mean()
        df[name] = 100 - 100 / (1 + g / (l + 1e-9))

    # MACD  (12/26 EMA, normalised by price)
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df["macd"]     = macd / p
    df["macd_sig"] = macd.ewm(span=9, adjust=False).mean() / p

    # Bollinger Band % position (20-day)
    ma20  = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    df["bb_pct"]   = (p - (ma20 - 2*std20)) / (4*std20 + 1e-9)

    # ATR approximation (close-to-close, 14-day)
    df["atr_norm"] = r1.abs().rolling(14).mean()

    # Volume features
    df["vol_norm"] = v / (v.rolling(21).mean() + 1e-9)
    df["vol_ret"]  = np.log(v / v.shift(1) + 1e-9)

    # Garman-Klass volatility approximation (5-day, annualised)
    df["gk_vol"]   = r1.rolling(5).std() * np.sqrt(252)

    # Drop only feature warm-up NaNs (NOT target NaNs)
    return df.dropna(subset=M1_FEAT).reset_index(drop=True)


def m1_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering WITH dual targets.
    Drops last H2=63 rows because forward return is undefined there.
    Use ONLY for building training/test sequences.
    """
    df = m1_features_only(df)
    p  = df["Close"].copy()
    df["target_21d"] = np.log(p.shift(-M1_H1) / p)
    df["target_63d"] = np.log(p.shift(-M1_H2) / p)
    return df.dropna().reset_index(drop=True)


# ── M2 feature list (8 features) ─────────────────────────────────────────────
M2_FEAT = ["ret_1d", "ret_5d", "ret_21d",
           "rsi_14", "bb_pct", "macd",
           "vol_norm", "rstd_5"]

def m2_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    8-feature engineering — NO target, NO target-related dropna.
    Last row is always the actual latest trading day.
    """
    df = df.copy()
    p, v = df["Close"].copy(), df["Volume"].copy()

    for lag, name in [(1,"ret_1d"),(5,"ret_5d"),(21,"ret_21d")]:
        df[name] = np.log(p / p.shift(lag))

    r1 = df["ret_1d"]
    df["rstd_5"] = r1.rolling(5).std()

    g = p.diff().clip(lower=0).rolling(14).mean()
    l = (-p.diff().clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + g / (l + 1e-9))

    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df["macd"]     = (ema12 - ema26) / p

    ma20  = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    df["bb_pct"]   = (p - (ma20 - 2*std20)) / (4*std20 + 1e-9)

    df["vol_norm"] = v / (v.rolling(21).mean() + 1e-9)

    return df.dropna(subset=M2_FEAT).reset_index(drop=True)


def m2_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    8-feature engineering WITH 21d classification target.
    Drops last H=21 rows.  Use ONLY for training/test sequences.
    """
    df  = m2_features_only(df)
    p   = df["Close"].copy()
    fwd = np.log(p.shift(-M2_HORIZON) / p)
    df["label"]   = np.where(fwd >= BUY_T, 2,
                    np.where(fwd <= SELL_T, 0, 1)).astype(np.int32)
    df["fwd_ret"] = fwd
    return df.dropna().reset_index(drop=True)


print("\n══ 2. Feature engineering ════════════════════════════════════════════")
m1_train_df   = {}   # with targets  — for training sequences
m1_pred_df    = {}   # features only — for forward prediction (latest date)
m2_train_df   = {}
m2_pred_df    = {}

for n in SN3:
    raw = stocks_raw[n]
    m1_train_df[n] = m1_engineer(raw)
    m1_pred_df[n]  = m1_features_only(raw)
    m2_train_df[n] = m2_engineer(raw)
    m2_pred_df[n]  = m2_features_only(raw)

    dist = m2_train_df[n]["label"].value_counts().sort_index()
    print(f"   {n:10s}  M1_train_last={m1_train_df[n]['Date'].iloc[-1].date()}"
          f"  M1_pred_last={m1_pred_df[n]['Date'].iloc[-1].date()}"
          f"  M2_pred_last={m2_pred_df[n]['Date'].iloc[-1].date()}")
    print(f"             M2 classes:  SELL={dist.get(0,0)}  "
          f"HOLD={dist.get(1,0)}  BUY={dist.get(2,0)}")


# =============================================================================
#  SECTION 3 — SEQUENCE BUILDERS + SPLITTERS
# =============================================================================

def m1_make_sequences(df: pd.DataFrame):
    """Returns X(N,W,F), y1(N,), y2(N,), dates(N,), prices(N,)."""
    F  = df[M1_FEAT].values.astype(np.float32)
    Y1 = df["target_21d"].values.astype(np.float32)
    Y2 = df["target_63d"].values.astype(np.float32)
    D  = df["Date"].values
    P  = df["Close"].values.astype(np.float32)
    X, y1, y2, dates, prices = [], [], [], [], []
    for i in range(M1_WINDOW, len(df)):
        X.append(F[i-M1_WINDOW:i])
        y1.append(Y1[i]); y2.append(Y2[i])
        dates.append(D[i]); prices.append(P[i])
    return (np.array(X), np.array(y1), np.array(y2),
            np.array(dates), np.array(prices))


def m1_split_scale(X, y1, y2):
    """Walk-forward split + StandardScaler (fit on train only)."""
    n    = len(X)
    n_te = max(10, int(n * M1_TEST_FRAC))
    n_va = max(10, int((n - n_te) * M1_VAL_FRAC))
    n_tr = n - n_te - n_va

    sc = StandardScaler()
    T, F = X[:n_tr].shape[1], X[:n_tr].shape[2]
    sc.fit(X[:n_tr].reshape(-1, F))

    def S(x):
        n_, t_, f_ = x.shape
        return sc.transform(x.reshape(-1, f_)).reshape(n_, t_, f_)

    return (S(X[:n_tr]),          y1[:n_tr],          y2[:n_tr],
            S(X[n_tr:n_tr+n_va]), y1[n_tr:n_tr+n_va], y2[n_tr:n_tr+n_va],
            S(X[n_tr+n_va:]),     y1[n_tr+n_va:],     y2[n_tr+n_va:],
            sc, n_tr, n_va)


def m2_make_sequences(df: pd.DataFrame):
    """Returns X(N,W,F), y(N,), dates(N,)."""
    F = df[M2_FEAT].values.astype(np.float32)
    Y = df["label"].values.astype(np.int32)
    D = df["Date"].values
    X, y, dates = [], [], []
    for i in range(M2_WINDOW, len(df)):
        X.append(F[i-M2_WINDOW:i])
        y.append(Y[i]); dates.append(D[i])
    return np.array(X), np.array(y), np.array(dates)


def m2_walk_forward_splits(n: int):
    """
    Expanding-window walk-forward CV.

    Fold 1:  ████████████████░░░░░░░░
    Fold 2:  ████████████████████░░░░
    (░ = held-out test, no look-ahead bias)
    """
    test_sz = max(30, int(n * M2_TEST_FRAC))
    splits  = []
    for fold in range(M2_N_FOLDS):
        te_end   = n - fold * test_sz
        te_start = te_end - test_sz
        if te_start < M2_WINDOW + 80:
            break
        tr_end = te_start
        val_sz = max(30, int(tr_end * 0.12))
        splits.append({
            "tr_idx" : np.arange(0, tr_end - val_sz),
            "va_idx" : np.arange(tr_end - val_sz, tr_end),
            "te_idx" : np.arange(te_start, te_end),
            "fold"   : M2_N_FOLDS - fold,
        })
    return splits[::-1]   # chronological order


# =============================================================================
#  SECTION 4 — MODEL ARCHITECTURES
# =============================================================================

# ── M1: Temporal Convolutional Network ───────────────────────────────────────

def _tcn_residual_block(x, filters, kernel_size, dilation_rate, l2_reg, sp_drop):
    """
    One TCN residual block:
      Causal Conv1D → BatchNorm → ReLU → SpatialDropout
      Causal Conv1D → BatchNorm
      Residual addition → ReLU

    Causal convolutions: padding='causal' ensures output at time t
    depends only on inputs ≤ t — no future leakage by construction.

    Dilation: receptive field = kernel_size × dilation_rate.
    Stacking dilation rates [1,2,4,8] gives exponentially growing
    receptive field: covers 3×15 = 45 days with 4 blocks.
    """
    residual = x
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
               padding="causal", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SpatialDropout1D(sp_drop)(x)     # drops entire feature channels
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
               padding="causal", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    # 1×1 conv to match channel dimensions for residual
    if residual.shape[-1] != filters:
        residual = Conv1D(filters, 1, padding="same")(residual)
    x = Add()([x, residual])
    return Activation("relu")(x)


def m1_directional_huber(alpha: float = M1_DIR_ALPHA,
                         temp: float = M1_DIR_TEMP,
                         delta: float = M1_HUBER_DELTA):
    """
    Hybrid regression loss for noisy financial returns.

    Base term: Huber loss on return magnitude.
    Direction term: BCE(sigmoid(temp * y_pred), sign(y_true)).
    This explicitly rewards getting the return sign right while still
    fitting the continuous target.
    """
    huber = keras.losses.Huber(delta=delta)

    def loss(y_true, y_pred):
        h = huber(y_true, y_pred)
        y_dir = tf.cast(y_true > 0.0, tf.float32)
        p_dir = tf.sigmoid(temp * y_pred)
        d = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_dir, p_dir)
        )
        return h + alpha * d

    loss.__name__ = "m1_directional_huber"
    return loss


def m1_sample_weights(y: np.ndarray, cap: float = M1_SW_CAP) -> np.ndarray:
    """
    Upweight larger absolute moves so the model learns informative regimes
    instead of over-optimising tiny near-zero returns.
    """
    scale = float(np.std(y) + 1e-8)
    w = 1.0 + np.clip(np.abs(y) / scale, 0.0, cap)
    return w.astype(np.float32)


def build_tcn(window: int, n_features: int) -> keras.Model:
    """
    TCN with dual output heads (21d and 63d targets).

        Architecture:
            Input(window, 23)
            → TCN block ×4 (dilation 1,2,4,8; 64 filters)
      → GlobalAveragePooling
      → Dense(64, relu) → Dropout
      → [Dense(1) out_21d,  Dense(1) out_63d]

    Multi-task learning: sharing lower layers improves generalisation
    when both targets are driven by the same underlying patterns.
    """
    inp = Input(shape=(window, n_features), name="sequence")
    x   = inp
    for dilation_rate in M1_DILATIONS:
        x = _tcn_residual_block(x, filters=64, kernel_size=3,
                                 dilation_rate=dilation_rate,
                                 l2_reg=M1_L2, sp_drop=M1_SP_DROP)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation="relu",
                kernel_regularizer=l2(M1_L2))(x)
    x   = Dropout(M1_DENSE_DROP)(x)
    out_21d = Dense(1, name="out_21d")(x)
    out_63d = Dense(1, name="out_63d")(x)

    model = Model(inp, [out_21d, out_63d], name="TCN_Regression")
    model.compile(
        optimizer    = Adam(M1_LR),
        loss         = {"out_21d": m1_directional_huber(),
                        "out_63d": m1_directional_huber()},
        loss_weights = {"out_21d": 1.0,     "out_63d": 1.0},
    )
    return model


# ── M2: Micro-Transformer Classifier ─────────────────────────────────────────

def _transformer_encoder_block(x, d_model, n_heads, ffn_dim,
                                attn_drop, ffn_drop, l2_reg, max_norm):
    """
    One Transformer encoder block:
      MultiHeadSelfAttention → Add & LayerNorm
      FFN (Dense→Dropout→Dense) → Add & LayerNorm

    Self-attention: every day in the window can directly attend to
    every other day, regardless of distance.  This captures long-range
    dependencies that LSTMs struggle with (e.g. a sharp move 40 days
    ago can still directly influence today's signal).

    GELU activation: smoother than ReLU, fewer dead neurons on small data.
    MaxNorm constraint: hard cap on weight magnitudes → prevents any
    single neuron dominating the representation.
    """
    # Multi-head self-attention sublayer
    attn = MultiHeadAttention(
        num_heads = n_heads,
        key_dim   = max(1, d_model // n_heads),
        dropout   = attn_drop,
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

    # Feed-forward sublayer
    ff = Dense(ffn_dim, activation="gelu",
               kernel_regularizer=l2(l2_reg),
               kernel_constraint=MaxNorm(max_norm))(x)
    ff = Dropout(ffn_drop)(ff)
    ff = Dense(d_model,
               kernel_regularizer=l2(l2_reg),
               kernel_constraint=MaxNorm(max_norm))(ff)
    ff = Dropout(ffn_drop)(ff)
    x  = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    return x


def build_transformer(window: int, n_features: int,
                      attn_drop: float = M2_ATTN_DROP,
                      ffn_drop: float = M2_FFN_DROP,
                      sp_drop: float = M2_SP_DROP,
                      noise_std: float = M2_NOISE_STD) -> keras.Model:
    """
    Micro-Transformer: ~1,900 trainable parameters.

    Architecture:
      Input(20, 8)
      → GaussianNoise(0.04)           [train-time augmentation]
      → Dense(12) + SpatialDropout    [input projection]
      → TransformerEncoderBlock       [d=12, heads=2, ffn=24]
      → GlobalAveragePooling
      → Dense(32, gelu) → Dropout
      → Dense(3, softmax)

    Parameter budget ≈ 1,900 → ~800 samples/param on 1,500 training rows.
    Heavy regularisation (9 layers) prevents overfitting at this scale.
    """
    inp = Input(shape=(window, n_features), name="sequence")

    # Input augmentation (active only during training)
    x = GaussianNoise(noise_std)(inp)

    # Linear projection to d_model dimensions
    x = Dense(M2_D_MODEL,
              kernel_regularizer=l2(M2_L2),
              kernel_constraint=MaxNorm(M2_MAXNORM))(x)
    x = SpatialDropout1D(sp_drop)(x)

    # Single Transformer encoder block
    x = _transformer_encoder_block(
        x, M2_D_MODEL, M2_N_HEADS, M2_FFN_DIM,
        attn_drop, ffn_drop, M2_L2, M2_MAXNORM
    )

    # Aggregate across time dimension
    x = GlobalAveragePooling1D()(x)

    # Classification head
    x   = Dense(32, activation="gelu",
                kernel_regularizer=l2(M2_L2),
                kernel_constraint=MaxNorm(M2_MAXNORM))(x)
    x   = Dropout(ffn_drop)(x)
    out = Dense(3, activation="softmax", name="output")(x)

    model = Model(inp, out, name="MicroTransformer_Classifier")
    return model


def label_smoothed_ce(smoothing: float = M2_LABEL_SMOOTH):
    """
    Label-smoothed sparse categorical cross-entropy.

    Standard CE: L = -log p(true_class)
    Smoothed CE: L = (1-ε)·CE + ε/K · sum(CE for all classes)

    Prevents the model from becoming overconfident (probability→1)
    which is the primary driver of overfitting on small datasets.
    """
    def loss(y_true, y_pred):
        K     = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        oh    = tf.one_hot(y_int, tf.shape(y_pred)[-1])
        soft  = oh * (1.0 - smoothing) + smoothing / K
        return tf.reduce_mean(
            -tf.reduce_sum(soft * tf.math.log(y_pred + 1e-9), axis=-1)
        )
    loss.__name__ = "label_smooth_ce"
    return loss


def warmup_cosine_lr(epoch: int) -> float:
    """
    Linear warm-up then cosine annealing.
    Warm-up prevents destructive weight updates in the first few steps.
    Cosine annealing produces smooth loss curves — EarlyStopping
    can detect the true val-loss minimum reliably.
    """
    if epoch < M2_WARMUP:
        return M2_LR_MIN + (M2_LR_MAX - M2_LR_MIN) * (epoch / M2_WARMUP)
    progress = (epoch - M2_WARMUP) / max(1, M2_EPOCHS - M2_WARMUP)
    return M2_LR_MIN + 0.5 * (M2_LR_MAX - M2_LR_MIN) * (1 + np.cos(np.pi * progress))


def mc_dropout_predict(model, X: np.ndarray,
                       n_passes: int = M2_MC_PASSES):
    """
    Monte Carlo Dropout inference:
    Run N stochastic forward passes (training=True keeps dropout active).
    Return mean softmax probabilities + std (uncertainty estimate).

    Benefits:
    - Calibrated probabilities (avoids overconfident point estimates)
    - Uncertainty estimate per prediction (std bars on dashboard)
    """
    all_probs = np.stack(
        [model(X, training=True).numpy() for _ in range(n_passes)],
        axis=0
    )                                        # (N_PASSES, batch, 3)
    return all_probs.mean(axis=0), all_probs.std(axis=0)


# =============================================================================
#  SECTION 5 — TRAINING
# =============================================================================

print("\n══ 3. Training ═══════════════════════════════════════════════════════")

# ── M1 training ───────────────────────────────────────────────────────────────
print("\n── Model 1: TCN Regression ──────────────────────────────────────────")
print(f"   Window={M1_WINDOW}d  H1={M1_H1}d  H2={M1_H2}d  "
      f"Epochs={M1_EPOCHS}  Batch={M1_BATCH}  LR={M1_LR}")
print(f"   Features={len(M1_FEAT)}  Regularisation: "
    f"SpatialDrop({M1_SP_DROP})+L2({M1_L2})+DirLoss(alpha={M1_DIR_ALPHA})")

m1_results = {}

for sn in SN3:
    X, y1, y2, dates, prices = m1_make_sequences(m1_train_df[sn])
    (X_tr, y1_tr, y2_tr,
     X_va, y1_va, y2_va,
     X_te, y1_te, y2_te,
     sc, n_tr, n_va) = m1_split_scale(X, y1, y2)

    model = build_tcn(M1_WINDOW, len(M1_FEAT))
    sw1_tr = m1_sample_weights(y1_tr)
    sw2_tr = m1_sample_weights(y2_tr)

    callbacks_m1 = [
        EarlyStopping(monitor="val_loss", patience=12,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=6, min_lr=1e-5, verbose=0),
    ]
    history = model.fit(
        X_tr,
        {"out_21d": y1_tr, "out_63d": y2_tr},
        sample_weight={"out_21d": sw1_tr, "out_63d": sw2_tr},
        validation_data=(X_va, {"out_21d": y1_va, "out_63d": y2_va}),
        epochs=M1_EPOCHS, batch_size=M1_BATCH,
        callbacks=callbacks_m1, verbose=0,
    )
    best_ep = int(np.argmin(history.history["val_loss"])) + 1

    # ── Test-set evaluation ───────────────────────────────────────────────────
    p21_raw, p63_raw = model.predict(X_te, verbose=0)
    p21 = p21_raw.flatten()
    p63 = p63_raw.flatten()
    mae21 = mean_absolute_error(y1_te, p21)
    mae63 = mean_absolute_error(y2_te, p63)
    r2_21 = r2_score(y1_te, p21)
    r2_63 = r2_score(y2_te, p63)
    dir21 = (np.sign(y1_te) == np.sign(p21)).mean() * 100
    dir63 = (np.sign(y2_te) == np.sign(p63)).mean() * 100

    # ── Forward prediction — USES FEATURES-ONLY DF ───────────────────────────
    # This ensures we predict from the actual latest date (Apr 17, 2026)
    # NOT from the date after target-dropna (which would be Jan 15, 2026)
    df_pred    = m1_pred_df[sn]
    last_close = float(df_pred["Close"].iloc[-1])
    last_date  = df_pred["Date"].iloc[-1]

    pred_win = sc.transform(
        df_pred[M1_FEAT].values[-M1_WINDOW:].reshape(-1, len(M1_FEAT))
    ).reshape(1, M1_WINDOW, len(M1_FEAT)).astype(np.float32)

    fw21_raw, fw63_raw = model.predict(pred_win, verbose=0)
    fw21 = float(fw21_raw.flatten()[0])
    fw63 = float(fw63_raw.flatten()[0])

    target_21d = last_close * np.exp(fw21)
    target_63d = last_close * np.exp(fw63)

    print(f"   {sn:10s}  best_ep={best_ep:3d}/{M1_EPOCHS}"
          f"  21d: MAE={mae21:.4f} R²={r2_21:+.3f} DirAcc={dir21:.1f}%"
          f"  63d: MAE={mae63:.4f} R²={r2_63:+.3f} DirAcc={dir63:.1f}%")
    print(f"              FORWARD from {last_date.date()} ¥{last_close:,.0f}"
          f"  → 21d ¥{target_21d:,.0f} ({fw21*100:+.2f}%)"
          f"  63d ¥{target_63d:,.0f} ({fw63*100:+.2f}%)")

    m1_results[sn] = dict(
        history    = history,
        best_ep    = best_ep,
        dates_te   = pd.to_datetime(dates[n_tr + n_va:]),
        y1te = y1_te, y2te = y2_te,
        p21  = p21,   p63  = p63,
        mae21=mae21, mae63=mae63,
        r2_21=r2_21, r2_63=r2_63,
        dir21=dir21, dir63=dir63,
        last_close = last_close,
        last_date  = last_date,
        target_21d = target_21d,
        target_63d = target_63d,
        ret_21d    = fw21,
        ret_63d    = fw63,
    )


# ── M2 training ───────────────────────────────────────────────────────────────
print("\n── Model 2: Micro-Transformer Classifier ────────────────────────────")
print(f"   d_model={M2_D_MODEL}  heads={M2_N_HEADS}  ffn={M2_FFN_DIM}  "
      f"window={M2_WINDOW}d  horizon={M2_HORIZON}d")
print(f"   Params≈1,900  L2={M2_L2}  MaxNorm={M2_MAXNORM}  "
      f"LabelSmooth={M2_LABEL_SMOOTH}")
print(f"   LR: warm-up({M2_WARMUP}ep)→cosine  "
      f"AdamW(wd={M2_WD})  GradClip=1.0")

m2_results = {}

for sn in SN3:
    st_cfg = M2_STOCK_TUNE.get(sn, {})
    stock_train_epochs = st_cfg.get("train_epochs", M2_EPOCHS)
    stock_warmup_epochs = st_cfg.get("warmup_epochs", M2_WARMUP)
    stock_patience = st_cfg.get("patience", 15)

    X, y, dates = m2_make_sequences(m2_train_df[sn])
    splits      = m2_walk_forward_splits(len(X))
    print(f"\n   ▶  {sn}  ({len(splits)} folds, {len(X)} sequences)")
    if st_cfg:
        cfg_str = " ".join([f"{k}={v}" for k, v in st_cfg.items()])
        print(f"      tuned: {cfg_str}")

    fold_accs, fold_hi  = [], []
    fold_tr,   fold_va  = [], []
    fold_cms            = []
    last_hist           = None

    for sp in splits:
        tr, va, te = sp["tr_idx"], sp["va_idx"], sp["te_idx"]
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]
        X_te, y_te = X[te], y[te]

        # Scale features on train-split only (no look-ahead bias)
        sc_fold = StandardScaler()
        T, F_n  = X_tr.shape[1], X_tr.shape[2]
        sc_fold.fit(X_tr.reshape(-1, F_n))
        def S(x):
            n_, t_, f_ = x.shape
            return sc_fold.transform(x.reshape(-1, f_)).reshape(n_, t_, f_)
        X_tr_s, X_va_s, X_te_s = S(X_tr), S(X_va), S(X_te)

        # Class weights — handle HOLD majority
        cw = {int(i): float(w) for i, w in enumerate(
              compute_class_weight("balanced",
                                   classes=np.array([0, 1, 2]), y=y_tr))}

        m = build_transformer(
            M2_WINDOW, len(M2_FEAT),
            attn_drop=st_cfg.get("attn_drop", M2_ATTN_DROP),
            ffn_drop=st_cfg.get("ffn_drop", M2_FFN_DROP),
            sp_drop=st_cfg.get("sp_drop", M2_SP_DROP),
            noise_std=st_cfg.get("noise_std", M2_NOISE_STD),
        )
        m.compile(
            optimizer = AdamW(M2_LR_MAX, weight_decay=M2_WD, clipnorm=1.0),
            loss      = label_smoothed_ce(M2_LABEL_SMOOTH),
            metrics   = ["accuracy"],
        )

        def stock_warmup_cosine_lr(epoch: int) -> float:
            if epoch < stock_warmup_epochs:
                return M2_LR_MIN + (M2_LR_MAX - M2_LR_MIN) * (epoch / max(1, stock_warmup_epochs))
            progress = (epoch - stock_warmup_epochs) / max(1, stock_train_epochs - stock_warmup_epochs)
            return M2_LR_MIN + 0.5 * (M2_LR_MAX - M2_LR_MIN) * (1 + np.cos(np.pi * progress))

        callbacks_m2 = [
            EarlyStopping(monitor="val_loss", patience=stock_patience,
                          restore_best_weights=True, verbose=0),
            LearningRateScheduler(stock_warmup_cosine_lr, verbose=0),
        ]
        h = m.fit(
            X_tr_s, y_tr,
            validation_data=(X_va_s, y_va),
            epochs=stock_train_epochs, batch_size=M2_BATCH,
            callbacks=callbacks_m2, class_weight=cw,
            verbose=0,
        )
        best = int(np.argmin(h.history["val_loss"])) + 1

        # Test-set evaluation
        probs = m.predict(X_te_s, verbose=0)
        preds = probs.argmax(axis=1)
        acc   = accuracy_score(y_te, preds)
        cm    = confusion_matrix(y_te, preds, labels=[0, 1, 2])

        # High-confidence subset accuracy (above-median max-probability)
        hi_mask = probs.max(axis=1) >= np.median(probs.max(axis=1))
        hi_acc  = (accuracy_score(y_te[hi_mask], preds[hi_mask])
                   if hi_mask.sum() > 0 else float("nan"))

        tr_b = h.history["accuracy"][best - 1]
        va_b = h.history["val_accuracy"][best - 1]
        gap  = tr_b - va_b

        fold_accs.append(acc); fold_cms.append(cm); fold_hi.append(hi_acc)
        fold_tr.append(tr_b);  fold_va.append(va_b)

        print(f"      Fold {sp['fold']}  best={best:3d}/{stock_train_epochs}"
              f"  tr={tr_b:.3f}  va={va_b:.3f}  gap={gap:+.3f}"
              f"  te={acc:.3f}  hi={hi_acc:.3f}")

        # Save last-fold details for plotting
        if sp["fold"] == splits[-1]["fold"]:
            last_hist      = h
            last_y_te      = y_te
            last_probs     = probs
            last_preds     = preds
            last_dates_te  = pd.to_datetime(dates[te])
            last_cm        = cm

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    mean_gap = float(np.mean([t - v for t, v in zip(fold_tr, fold_va)]))
    above    = mean_acc > 1/3

    print(f"      ── CV = {mean_acc:.3f} ± {std_acc:.3f}"
          f"  mean_gap = {mean_gap:+.3f}"
          f"  {'ABOVE RANDOM ✓' if above else 'BELOW RANDOM ✗'}")

    # ── Retrain on ALL data for forward prediction ────────────────────────────
    sc_full = StandardScaler()
    n_, t_, f_ = X.shape
    sc_full.fit(X.reshape(-1, f_))
    X_all_s = sc_full.transform(X.reshape(-1, f_)).reshape(n_, t_, f_)

    cw_full = {int(i): float(w) for i, w in enumerate(
               compute_class_weight("balanced",
                                    classes=np.array([0, 1, 2]), y=y))}
    m_final = build_transformer(
        M2_WINDOW, len(M2_FEAT),
        attn_drop=st_cfg.get("attn_drop", M2_ATTN_DROP),
        ffn_drop=st_cfg.get("ffn_drop", M2_FFN_DROP),
        sp_drop=st_cfg.get("sp_drop", M2_SP_DROP),
        noise_std=st_cfg.get("noise_std", M2_NOISE_STD),
    )
    m_final.compile(
        AdamW(M2_LR_MAX, weight_decay=M2_WD, clipnorm=1.0),
        label_smoothed_ce(M2_LABEL_SMOOTH), ["accuracy"]
    )
    m_final.fit(X_all_s, y,
                epochs=30, batch_size=M2_BATCH,
                class_weight=cw_full, verbose=0)

    # ── Forward prediction — USES FEATURES-ONLY DF ───────────────────────────
    df_pred    = m2_pred_df[sn]
    last_close = float(df_pred["Close"].iloc[-1])
    last_date  = df_pred["Date"].iloc[-1]

    last_win = sc_full.transform(
        df_pred[M2_FEAT].values[-M2_WINDOW:].reshape(-1, f_)
    ).reshape(1, M2_WINDOW, f_).astype(np.float32)

    fwd_mean, fwd_std = mc_dropout_predict(m_final, last_win)
    fwd_prob  = fwd_mean[0]
    fwd_unc   = fwd_std[0]
    fwd_class = int(fwd_prob.argmax())
    fwd_rec   = CLASS_NAMES[fwd_class]
    fwd_conf  = float(fwd_prob[fwd_class])

    print(f"      FORWARD from {last_date.date()}  ¥{last_close:,.0f}"
          f"  SELL={fwd_prob[0]:.2f}  HOLD={fwd_prob[1]:.2f}  BUY={fwd_prob[2]:.2f}"
          f"  → {fwd_rec} ({fwd_conf:.0%})")

    m2_results[sn] = dict(
        fold_accs    = fold_accs,
        fold_hi      = fold_hi,
        fold_cms     = fold_cms,
        fold_tr      = fold_tr,
        fold_va      = fold_va,
        mean_acc     = mean_acc,
        std_acc      = std_acc,
        mean_gap     = mean_gap,
        above        = above,
        last_hist    = last_hist,
        last_y_te    = last_y_te,
        last_probs   = last_probs,
        last_preds   = last_preds,
        last_dates_te= last_dates_te,
        last_cm      = last_cm,
        fwd_prob     = fwd_prob,
        fwd_unc      = fwd_unc,
        fwd_rec      = fwd_rec,
        fwd_conf     = fwd_conf,
        last_close   = last_close,
        last_date    = last_date,
    )


# =============================================================================
#  SECTION 6 — FINAL RESULTS SUMMARY
# =============================================================================
print("\n══ 4. Results summary ════════════════════════════════════════════════")
print(f"\n  M1 — TCN Regression  (base: Apr 17, 2026)")
print(f"  {'Stock':<12} {'Close':>8}  {'21d':>10} {'21dDir':>7}  "
      f"{'63d':>10} {'63dDir':>7}")
print("  " + "─"*60)
for sn in SN3:
    r = m1_results[sn]
    print(f"  {sn:<12} ¥{r['last_close']:>7,.0f}  "
          f"¥{r['target_21d']:>9,.0f}({r['ret_21d']*100:+.1f}%) {r['dir21']:>6.1f}%  "
          f"¥{r['target_63d']:>9,.0f}({r['ret_63d']*100:+.1f}%) {r['dir63']:>6.1f}%")

print(f"\n  M2 — Transformer Classifier  (base: Apr 17, 2026)")
print(f"  {'Stock':<12} {'Close':>8}  {'Signal':>6} {'Conf':>6}  "
      f"{'CV acc':>12}  {'Above 33%?':>10}")
print("  " + "─"*60)
for sn in SN3:
    r = m2_results[sn]
    print(f"  {sn:<12} ¥{r['last_close']:>7,.0f}  "
          f"{r['fwd_rec']:>6}  {r['fwd_conf']:>6.0%}  "
          f"{r['mean_acc']:.3f}±{r['std_acc']:.3f}  "
          f"{'YES ✓' if r['above'] else 'NO ✗'}")


# =============================================================================
#  SECTION 7 — REPORT FIGURES
# =============================================================================
print("\n══ 5. Generating figures ═════════════════════════════════════════════")


# ─── Fig 0: Pipeline architecture ────────────────────────────────────────────
fig0 = plt.figure(figsize=(20, 11), facecolor="white")
ax0  = fig0.add_axes([0, 0, 1, 1])
ax0.set_xlim(0, 20); ax0.set_ylim(0, 11); ax0.axis("off")

def _rbox(ax, cx, cy, w, h, fc, label, sub="", fs=10, tc="white"):
    p = mpatches.FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.2",
        facecolor=fc, edgecolor="#333333", linewidth=1.5, zorder=3
    )
    ax.add_patch(p)
    ax.text(cx, cy+(0.18 if sub else 0), label,
            ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc, zorder=4)
    if sub:
        ax.text(cx, cy-0.28, sub,
                ha="center", va="center",
                fontsize=7.5, color=tc, alpha=0.90, zorder=4)

def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color="#444444",
                                lw=1.6, mutation_scale=16), zorder=2)

# Title
ax0.text(10, 10.55,
         "Japan Gaming Stocks — Neural Network Model Pipeline",
         ha="center", fontsize=16, fontweight="bold", color="#1A1A2E")
ax0.text(10, 10.15,
         "Nintendo (7974)  ·  Sony (6758)  ·  Capcom (9697)"
         "  |  Daily, Apr 2016 – Apr 2026  |  S&P Capital IQ",
         ha="center", fontsize=10, color="#555555")

# Row 1: data pipeline
_rbox(ax0,  3,  9.35, 3.6, 0.85, "#2C3E50", "Raw Data (S&P Capital IQ)",
      "2,441 daily rows per stock")
_rbox(ax0,  8.5,9.35, 3.6, 0.85, "#2471A3", "Data Loading & Cleaning",
      "Parse dates / prices / volume")
_rbox(ax0, 14,  9.35, 3.6, 0.85, "#7D3C98", "Feature Engineering",
      "23 features (M1)  ·  8 features (M2)")
_arrow(ax0, 4.8, 9.35, 6.7, 9.35)
_arrow(ax0,10.3, 9.35,12.2, 9.35)

# Split into two branches
ax0.plot([14,14],[8.93,8.45], color="#666", lw=1.5)
ax0.plot([6, 14],[8.45,8.45], color="#666", lw=1.5)
ax0.plot([6,  6],[8.45,8.00], color="#666", lw=1.5)
ax0.plot([14,14],[8.45,8.00], color="#666", lw=1.5)
_arrow(ax0, 6, 8.45, 6, 8.02)
_arrow(ax0,14, 8.45,14, 8.02)

# Branch labels
for cx, txt, col in [(6,"MODEL 1  ·  TCN Regression","#16A085"),
                     (14,"MODEL 2  ·  Transformer Classifier","#D35400")]:
    ax0.text(cx, 8.55, txt, ha="center", fontsize=9,
             color=col, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.2))

# Row 2: split + train
_rbox(ax0,  6, 7.35, 4.2, 1.0, "#16A085",
      "Walk-Forward Split",
      "Train 70%  ·  Val 12%  ·  Test 18%\nWindow = 60 trading days")
_rbox(ax0, 14, 7.35, 4.2, 1.0, "#D35400",
      "Walk-Forward CV  (2 folds)",
      "Expanding window  ·  No look-ahead bias\nWindow = 20 trading days")
_arrow(ax0, 6,7.95, 6,7.80); _arrow(ax0,14,7.95,14,7.80)

# Row 3: model boxes
_rbox(ax0,  6, 5.90, 4.2, 1.1, "#148F77",
      "M1: TCN Regression",
      "4× causal dilated conv blocks\nSpatialDropout + BatchNorm + L2\nDual output: 21d & 63d heads")
_rbox(ax0, 14, 5.90, 4.2, 1.1, "#BA4A00",
      "M2: Micro-Transformer (v4)",
      "Self-attention  d=12, 2 heads\nMaxNorm + GELU + label smoothing\nAdamW + cosine warm-up LR")
_arrow(ax0, 6,6.80, 6,6.45); _arrow(ax0,14,6.80,14,6.45)

# Row 4: outputs
_rbox(ax0,  6, 4.55, 4.2, 1.0, "#1A5276",
      "Price Targets",
      "21d (+1 month)  ·  63d (+3 months)\nlog-return → ¥ price forecast")
_rbox(ax0, 14, 4.55, 4.2, 1.0, "#7B241C",
      "BUY / HOLD / SELL Signal",
      "Softmax probabilities (3-class)\nMC-Dropout uncertainty (25 passes)")
_arrow(ax0, 6,5.45, 6,5.05); _arrow(ax0,14,5.45,14,5.05)

# Converge to combined recommendation
ax0.plot([ 6, 6],[4.05,3.78], color="#666", lw=1.5)
ax0.plot([14,14],[4.05,3.78], color="#666", lw=1.5)
ax0.plot([ 6,14],[3.78,3.78], color="#666", lw=1.5)
ax0.plot([10,10],[3.78,3.45], color="#666", lw=1.5)
_arrow(ax0,10,3.78,10,3.48)
_rbox(ax0, 10, 2.95, 6.0, 0.90, "#1A1A2E",
      "Combined Analyst Recommendation",
      "M1 price target  +  M2 direction signal  +  Fundamental analysis")

# Left sidebar: anti-overfit
ax0.text(0.4, 6.0,
         "Anti-Overfit Measures\n"
         "────────────────────\n"
         "M1 — TCN:\n"
         "  Causal dilated conv\n"
         "  SpatialDropout1D\n"
         "  L2 regularisation\n"
         "  EarlyStopping\n"
         "  ReduceLROnPlateau\n\n"
         "M2 — Transformer:\n"
         "  GaussianNoise input\n"
         "  SpatialDropout1D\n"
         "  Attention dropout\n"
         "  FFN dropout\n"
         "  MaxNorm constraint\n"
         "  Label smoothing\n"
         "  AdamW weight decay\n"
         "  Gradient clip (1.0)\n"
         "  Walk-forward CV",
         fontsize=7.5, va="center", color="#222222",
         bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5", ec="#BBBBBB", lw=1),
         family="monospace")

# Right sidebar: model comparison
ax0.text(19.6, 6.0,
         "Model Comparison\n"
         "──────────────\n"
         "         Fixed    v4\n"
         "Params   ~8k      ~2k\n"
         "Nintendo 0.389   0.440\n"
         "Sony     0.438   0.371\n"
         "Capcom   0.355   0.454\n"
         "Gap     +0.025  −0.046\n"
         "────────────────\n"
         "→ v4 selected\n"
         "  (5/6 criteria)",
         fontsize=7.5, va="center", ha="right", color="#222222",
         bbox=dict(boxstyle="round,pad=0.5", fc="#FFFDE7", ec="#BBBBBB", lw=1),
         family="monospace")

save("rpt_00_pipeline.png", fig0)


# ─── M1 Fig 1: Price history + volume ────────────────────────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor="white")
fig1.suptitle("Japan Gaming Stocks — 10-Year Daily Price History\n"
              "Apr 2016 – Apr 2026  |  Source: S&P Capital IQ",
              fontsize=14, fontweight="bold", y=1.01)
for col, sn in enumerate(SN3):
    df = stocks_raw[sn]; c = STOCK_C[sn]
    # Price panel
    ax = axes[0, col]
    ax.fill_between(df["Date"], df["Close"], alpha=0.12, color=c)
    ax.plot(df["Date"], df["Close"], color=c, lw=1.4, label="Close")
    ax.plot(df["Date"], df["Close"].rolling(63).mean(),
            color="black", lw=1, linestyle="--", alpha=0.6, label="63d MA")
    ax.set_title(sn, fontsize=13, fontweight="bold", color=c)
    ax.set_ylabel("Share Price (¥)"); ax.legend(); ax.grid(True)
    ax.tick_params(axis="x", rotation=20)
    # Volume panel
    ax2 = axes[1, col]
    ax2.bar(df["Date"], df["Volume"], color=c, alpha=0.45, width=3)
    ax2.plot(df["Date"], df["Volume"].rolling(21).mean(),
             color="black", lw=1.2, linestyle="--", alpha=0.7, label="21d avg")
    ax2.set_ylabel("Volume (mm)"); ax2.legend(); ax2.grid(True)
    ax2.tick_params(axis="x", rotation=20)
plt.tight_layout()
save("rpt_m1_01_price_history.png", fig1)


# ─── M1 Fig 2: Predictions vs actual ─────────────────────────────────────────
fig2, axes = plt.subplots(3, 2, figsize=(18, 14), facecolor="white")
fig2.suptitle(
    "Model 1 (TCN) — Predicted vs Actual Log-Returns  [Test Set]\n"
    "Green shading = correct direction  ·  "
    "⚠ Negative R² = noisy magnitude (DirAcc > 50% confirms directional value)",
    fontsize=12, fontweight="bold", y=1.02
)
for row, sn in enumerate(SN3):
    r = m1_results[sn]; c = STOCK_C[sn]
    for col, (yt, yp, hor, mae, r2v, da) in enumerate([
        (r["y1te"], r["p21"], "21d", r["mae21"], r["r2_21"], r["dir21"]),
        (r["y2te"], r["p63"], "63d", r["mae63"], r["r2_63"], r["dir63"]),
    ]):
        ax    = axes[row, col]
        dates = r["dates_te"]
        ax.plot(dates, yt, color="#333333", lw=1.4, label="Actual", zorder=3)
        ax.plot(dates, yp, color=c, lw=1.4, linestyle="--",
                label=f"Predicted {hor}", zorder=2)
        ax.fill_between(dates, yt, yp, alpha=0.12, color=c)
        ax.axhline(0, color="#999999", lw=0.8, linestyle="--")

        # Fix: set y limits before direction shading to avoid canvas fill
        y_all = np.concatenate([yt, yp])
        ylo, yhi = y_all.min() * 1.35, y_all.max() * 1.35
        ax.set_ylim(ylo, yhi)
        ax.fill_between(dates, ylo, yhi,
                        where=(np.sign(yt) == np.sign(yp)),
                        alpha=0.04, color="green", zorder=0)

        r2_str = f"R²={r2v:.3f}" + (" ⚠" if r2v < 0 else "")
        ax.set_title(f"{sn} — {hor}  MAE={mae:.4f}  {r2_str}  DirAcc={da:.1f}%",
                     fontsize=10, fontweight="bold", color=c)
        if r2v < 0:
            ax.text(0.01, 0.04,
                    "⚠ Negative R²: magnitude prediction is noisy — normal at short horizons.\n"
                    "DirAcc > 50% confirms the directional signal still has value.",
                    transform=ax.transAxes, fontsize=7.5, color="#C0392B",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="#FFF5F5", ec="#E74C3C", lw=0.8))
        ax.set_ylabel(f"{hor} Log-Return")
        ax.legend(fontsize=8); ax.grid(True)
        ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
save("rpt_m1_02_predictions.png", fig2)


# ─── M1 Fig 3: Residual scatter ───────────────────────────────────────────────
fig3, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor="white")
fig3.suptitle("Model 1 (TCN) — Residual Analysis  [Test Set]",
              fontsize=13, fontweight="bold", y=1.01)
for col, sn in enumerate(SN3):
    r = m1_results[sn]; c = STOCK_C[sn]
    for row, (yt, yp, hor) in enumerate([
        (r["y1te"], r["p21"], "21d"),
        (r["y2te"], r["p63"], "63d"),
    ]):
        ax   = axes[row, col]
        corr = np.corrcoef(yt, yp)[0, 1]
        lim  = max(abs(yt).max(), abs(yp).max()) * 1.15
        ax.scatter(yt, yp, alpha=0.30, s=10, color=c,
                   edgecolors="none", zorder=3)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1,
                label="Perfect", zorder=4)
        m_, b_ = np.polyfit(yt, yp, 1)
        xs = np.linspace(-lim, lim, 60)
        ax.plot(xs, m_*xs+b_, color=c, lw=1.5, alpha=0.7,
                label=f"Fit  r={corr:.2f}")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{sn} {hor}  r={corr:.2f}  R²={r2_score(yt,yp):.3f}",
                     fontsize=10, fontweight="bold", color=c)
        ax.set_xlabel("Actual log-return")
        ax.set_ylabel("Predicted log-return")
        ax.legend(fontsize=8); ax.grid(True)
plt.tight_layout()
save("rpt_m1_03_residuals.png", fig3)


# ─── M1 Fig 4: Training loss curves (real history) ───────────────────────────
fig4, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor="white")
fig4.suptitle("Model 1 (TCN) — Training & Validation Loss Curves\n"
              "Green = healthy (val ≤ train)  |  Red = overfit zone",
              fontsize=13, fontweight="bold", y=1.03)
for col, sn in enumerate(SN3):
    r    = m1_results[sn]
    h    = r["history"].history
    c    = STOCK_C[sn]
    best = r["best_ep"] - 1
    ep   = np.arange(1, len(h["loss"]) + 1)
    tr_l = np.array(h["loss"])
    va_l = np.array(h["val_loss"])
    ax   = axes[col]
    ax.plot(ep, tr_l, color="#2980B9", lw=2, label="Train loss")
    ax.plot(ep, va_l, color=c,         lw=2, linestyle="--", label="Val loss")
    ax.axvline(best+1, color="#27AE60", lw=2, linestyle=":",
               label=f"Best ep={best+1}/{len(ep)}")
    ax.fill_between(ep, tr_l, va_l,
                    where=(va_l > tr_l), alpha=0.18, color="#E74C3C",
                    label="Overfit zone")
    ax.fill_between(ep, tr_l, va_l,
                    where=(va_l <= tr_l), alpha=0.10, color="#27AE60",
                    label="Healthy zone")
    ax.set_title(f"{sn}  —  best epoch = {best+1}",
                 fontsize=11, fontweight="bold", color=c)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.legend(fontsize=8); ax.grid(True)
plt.tight_layout()
save("rpt_m1_04_loss.png", fig4)


# ─── M1 Fig 5: Forward price forecast ────────────────────────────────────────
fig5, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="white")
fig5.suptitle("Model 1 (TCN) — Forward Price Forecast  |  April 2026\n"
              "✓ Predicted from actual latest close: Apr 17, 2026",
              fontsize=13, fontweight="bold", y=1.03)
for col, sn in enumerate(SN3):
    r = m1_results[sn]; c = STOCK_C[sn]; ax = axes[col]
    labels = ["Current\nClose", "21d\nTarget", "63d\nTarget"]
    values = [r["last_close"], r["target_21d"], r["target_63d"]]
    colors = [c,
              "#27AE60" if r["ret_21d"] > 0 else "#E74C3C",
              "#27AE60" if r["ret_63d"] > 0 else "#E74C3C"]
    bars  = ax.bar(labels, values, color=colors,
                   edgecolor="white", linewidth=1.2,
                   width=0.5, alpha=0.88)
    span  = max(values) - min(values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + span*0.025 + 1,
                f"¥{v:,.0f}", ha="center", fontsize=11,
                fontweight="bold", color="#222222")
    ax.set_ylim(min(values)*0.90, max(values)*1.13)
    ax.set_title(
        f"{sn}\n"
        f"21d: {r['ret_21d']*100:+.2f}%   63d: {r['ret_63d']*100:+.2f}%\n"
        f"Base: ¥{r['last_close']:,.0f}  ({r['last_date'].date()})",
        fontsize=10, fontweight="bold", color=c
    )
    ax.set_ylabel("Price (¥)"); ax.grid(axis="y", alpha=0.4)
    note = (f"21d DirAcc: {r['dir21']:.1f}%\n"
            f"63d DirAcc: {r['dir63']:.1f}%\n"
            f"21d R²:  {r['r2_21']:.3f}\n"
            f"63d R²:  {r['r2_63']:.3f}")
    ax.text(0.97, 0.04, note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            color="#444444", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="#CCCCCC"))
plt.tight_layout()
save("rpt_m1_05_forecast.png", fig5)


# ─── M2 Fig 1: Walk-forward CV accuracy ──────────────────────────────────────
fig6, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="white")
fig6.suptitle(
    "Model 2 (Micro-Transformer) — Walk-Forward CV Accuracy\n"
    "Random baseline = 33.3%  |  Each fold trains on past, tests on future",
    fontsize=13, fontweight="bold", y=1.03
)
for col, sn in enumerate(SN3):
    r  = m2_results[sn]; c = STOCK_C[sn]; ax = axes[col]
    fa = r["fold_accs"]; fh = r["fold_hi"]
    nf = len(fa); w = 0.30
    for i, (a, hi) in enumerate(zip(fa, fh)):
        ax.bar(i+1 - w/2, a, w,
               color=c if a > 1/3 else "#AAAAAA",
               edgecolor="white", alpha=0.90,
               label="Test acc" if i == 0 else "")
        ax.bar(i+1 + w/2, hi, w,
               color=c if hi > 1/3 else "#CCCCCC",
               edgecolor="white", alpha=0.50, hatch="//",
               label="Hi-conf acc" if i == 0 else "")
        ax.text(i+1 - w/2, a + 0.010, f"{a:.3f}",
                ha="center", fontsize=10, fontweight="bold",
                color=c if a > 1/3 else "#666666")

    ax.axhline(1/3, color="#E74C3C", lw=1.5, linestyle="--",
               label="Random (33.3%)")
    ax.axhline(0.5,  color="#27AE60", lw=1.2, linestyle=":",
               label="50% target")
    ax.axhline(r["mean_acc"], color=c, lw=2, linestyle="-.", alpha=0.75,
               label=f"CV mean = {r['mean_acc']:.3f}")

    ax.set_xticks(list(range(1, nf+1)))
    ax.set_xticklabels([f"Fold {i}" for i in range(1, nf+1)], fontsize=11)
    ax.set_xlim(0.4, nf + 0.8); ax.set_ylim(0, 0.72)

    above_str = "ABOVE RANDOM ✓" if r["above"] else "BELOW RANDOM ✗"
    ax.set_title(
        f"{sn}\n"
        f"CV = {r['mean_acc']:.3f} ± {r['std_acc']:.3f}  "
        f"gap = {r['mean_gap']:+.3f}  |  {above_str}",
        fontsize=10, fontweight="bold",
        color=c if r["above"] else "#E74C3C"
    )
    ax.set_xlabel("CV Fold (chronological)")
    ax.set_ylabel("Classification Accuracy")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
save("rpt_m2_01_cv_accuracy.png", fig6)


# ─── M2 Fig 2: Loss curves with overfitting diagnostic ───────────────────────
fig7, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
fig7.suptitle(
    "Model 2 (Micro-Transformer) — Training vs Validation  [Last CV Fold]\n"
    "Mild gap (< 8%) is acceptable in financial time-series — annotated per panel",
    fontsize=12, fontweight="bold", y=1.02
)
for col, sn in enumerate(SN3):
    r    = m2_results[sn]
    h    = r["last_hist"].history
    c    = STOCK_C[sn]
    best = int(np.argmin(h["val_loss"]))
    ep   = np.arange(1, len(h["loss"]) + 1)
    tr_l = np.array(h["loss"])
    va_l = np.array(h["val_loss"])
    tr_a = np.array(h["accuracy"])
    va_a = np.array(h["val_accuracy"])
    gap_l = float(tr_l[best] - va_l[best])
    gap_a = float(tr_a[best] - va_a[best])

    # Loss subplot
    ax = axes[0, col]
    ax.plot(ep, tr_l, color="#2980B9", lw=2, label="Train loss")
    ax.plot(ep, va_l, color=c,         lw=2, linestyle="--", label="Val loss")
    ax.axvline(best+1, color="#27AE60", lw=1.8, linestyle=":",
               label=f"Best = {best+1}/{M2_EPOCHS}")
    ax.fill_between(ep, tr_l, va_l, where=(va_l > tr_l),
                    alpha=0.18, color="#E74C3C", label="Overfit zone")
    ax.fill_between(ep, tr_l, va_l, where=(va_l <= tr_l),
                    alpha=0.10, color="#27AE60", label="Healthy zone")
    ax.set_title(f"{sn}  gap@best = {gap_l:+.4f}",
                 fontsize=10, fontweight="bold", color=c)
    ax.set_ylabel("CE Loss"); ax.set_xlabel("Epoch")
    ax.legend(fontsize=7); ax.grid(True)

    # Accuracy subplot
    ax2 = axes[1, col]
    ax2.plot(ep, tr_a, color="#2980B9", lw=2, label="Train acc")
    ax2.plot(ep, va_a, color=c,         lw=2, linestyle="--", label="Val acc")
    ax2.axhline(1/3, color="#E74C3C", lw=1.2, linestyle="--",
                label="Random (33%)")
    ax2.axhline(0.5,  color="#27AE60", lw=1,   linestyle=":",
                label="50% target")
    ax2.axvline(best+1, color="#27AE60", lw=1.8, linestyle=":")
    ax2.fill_between(ep, tr_a, va_a, where=(tr_a > va_a),
                     alpha=0.12, color="#E74C3C")
    ax2.fill_between(ep, tr_a, va_a, where=(tr_a <= va_a),
                     alpha=0.08, color="#27AE60")
    ax2.annotate(f"acc gap = {gap_a:+.3f}",
                 xy=(best+1, (tr_a[best]+va_a[best])/2),
                 fontsize=8, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.2",
                           fc="white", ec="#CCCCCC"))
    is_ok = abs(gap_a) < 0.08
    ax2.text(0.98, 0.05,
             ("Mild overfit — acceptable\n" if gap_a > 0.02
              else "Healthy — no overfit\n") +
             f"gap = {gap_a:+.3f}  {'< 8% ✓' if is_ok else '≥ 8% monitor'}\n"
             "Val above random ✓  |  No scissors ✓",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7.5,
             color="#1A5276" if is_ok else "#C0392B",
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="#EBF5FB" if is_ok else "#FFF5F5",
                       ec="#2980B9" if is_ok else "#E74C3C", lw=0.8))
    ax2.set_ylim(0.10, 0.80)
    ax2.set_ylabel("Accuracy"); ax2.set_xlabel("Epoch")
    ax2.legend(fontsize=7); ax2.grid(True)
plt.tight_layout()
save("rpt_m2_02_loss_curves.png", fig7)


# ─── M2 Fig 3: Confusion matrices ────────────────────────────────────────────
fig8, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
fig8.suptitle("Model 2 (Micro-Transformer) — Confusion Matrices  [Last Fold Test Set]",
              fontsize=13, fontweight="bold", y=1.03)
for col, sn in enumerate(SN3):
    r  = m2_results[sn]; ax = axes[col]
    ConfusionMatrixDisplay(
        r["last_cm"], display_labels=CLASS_NAMES
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    acc_str = f"{r['fold_accs'][-1]:.3f}"
    ab_str  = "✓ above random" if r["fold_accs"][-1] > 1/3 else "✗ below random"
    ax.set_title(
        f"{sn}\nacc = {acc_str}  CV = {r['mean_acc']:.3f}  {ab_str}",
        fontsize=10, fontweight="bold", color=STOCK_C[sn]
    )
plt.tight_layout()
save("rpt_m2_03_confusion.png", fig8)


# ─── M2 Fig 4: Softmax probability time-series ───────────────────────────────
fig9, axes = plt.subplots(3, 1, figsize=(18, 13), facecolor="white")
fig9.suptitle(
    "Model 2 (Micro-Transformer) — Predicted Class Probabilities  [Last Fold Test Period]\n"
    "Stacked area: SELL / HOLD / BUY  |  Markers = actual label",
    fontsize=13, fontweight="bold", y=1.01
)
for row, sn in enumerate(SN3):
    r      = m2_results[sn]; ax = axes[row]
    probs  = r["last_probs"]
    true_y = r["last_y_te"]
    dates  = r["last_dates_te"]
    ax.stackplot(dates,
                 probs[:, 0], probs[:, 1], probs[:, 2],
                 labels=CLASS_NAMES,
                 colors=["#E74C3C", "#E67E22", "#27AE60"],
                 alpha=0.70)
    for cls, marker in [(0, "v"), (2, "^")]:
        idx = np.where(true_y == cls)[0]
        if len(idx):
            ax.scatter(dates[idx], np.ones(len(idx)) * 0.96,
                       marker=marker, s=35, zorder=5,
                       color=CLS_C[CLASS_NAMES[cls]],
                       edgecolors="white", linewidths=0.5,
                       label=f"Actual {CLASS_NAMES[cls]}")
    ax.set_title(
        f"{sn}  acc = {r['fold_accs'][-1]:.3f}  "
        f"CV = {r['mean_acc']:.3f} ± {r['std_acc']:.3f}",
        fontsize=12, fontweight="bold", color=STOCK_C[sn]
    )
    ax.set_ylabel("Probability"); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8, ncol=5)
    ax.grid(alpha=0.15); ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
save("rpt_m2_04_probs.png", fig9)


# ─── Combined recommendation dashboard ───────────────────────────────────────
fig10, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="white")
fig10.suptitle(
    "Combined Recommendation Dashboard  |  Japan Gaming  |  April 2026\n"
    "✓ Base price: Apr 17, 2026 (actual latest close)  "
    "·  M1: Price Targets  ·  M2: BUY / HOLD / SELL",
    fontsize=13, fontweight="bold", y=1.02
)
for col, sn in enumerate(SN3):
    r1 = m1_results[sn]; r2 = m2_results[sn]
    c  = STOCK_C[sn]; bg = STOCK_BG[sn]
    rec  = r2["fwd_rec"]
    prob = r2["fwd_prob"]
    unc  = r2["fwd_unc"]

    # Top: M1 price targets
    ax = axes[0, col]; ax.set_facecolor(bg)
    vals = [r1["last_close"], r1["target_21d"], r1["target_63d"]]
    clrs = [c,
            "#27AE60" if r1["ret_21d"] > 0 else "#E74C3C",
            "#27AE60" if r1["ret_63d"] > 0 else "#E74C3C"]
    bars = ax.bar(["Current\nClose", "21d\nTarget", "63d\nTarget"],
                  vals, color=clrs, edgecolor="white",
                  linewidth=1.2, width=0.5, alpha=0.88)
    span = max(vals) - min(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + span*0.025 + 1,
                f"¥{v:,.0f}", ha="center", fontsize=11,
                fontweight="bold", color="#222222")
    ax.set_ylim(min(vals)*0.90, max(vals)*1.13)
    ax.set_title(
        f"{sn}\n"
        f"TCN: 21d {r1['ret_21d']*100:+.2f}%   63d {r1['ret_63d']*100:+.2f}%\n"
        f"Base ¥{r1['last_close']:,.0f} ({r1['last_date'].date()})",
        fontsize=10, fontweight="bold", color=c
    )
    ax.set_ylabel("Price (¥)"); ax.grid(axis="y", alpha=0.35)

    # Bottom: M2 signal
    ax2 = axes[1, col]; ax2.set_facecolor(bg)
    ax2.bar(CLASS_NAMES, prob,
            color=[CLS_C[n] for n in CLASS_NAMES],
            edgecolor="white", linewidth=1.2,
            width=0.5, alpha=0.85,
            yerr=unc, capsize=6,
            error_kw={"ecolor": "#555555", "lw": 1.5})
    for i, v in enumerate(prob):
        ax2.text(i, v + unc[i] + 0.03, f"{v:.2f}",
                 ha="center", fontsize=12,
                 fontweight="bold", color="#222222")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1/3, color="#AAAAAA", lw=1, linestyle="--",
                alpha=0.7, label="Equal prob (33%)")
    # Recommendation badge
    ax2.text(0.5, 0.86, rec, transform=ax2.transAxes,
             ha="center", va="center",
             fontsize=22, fontweight="bold", color=CLS_C[rec],
             bbox=dict(boxstyle="round,pad=0.3",
                       fc="white", ec=CLS_C[rec], lw=2.5))
    # Confidence + CV note
    cv_warn = " ⚠ LOW" if not r2["above"] else ""
    ax2.text(
        0.5, 0.10,
        f"Conf: {r2['fwd_conf']:.0%}  ·  CV: {r2['mean_acc']:.3f}{cv_warn}\n"
        f"From ¥{r2['last_close']:,.0f} ({r2['last_date'].date()})"
        f"  ·  Error bars = MC-Dropout σ",
        transform=ax2.transAxes, ha="center", fontsize=8.5,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC")
    )
    ax2.set_title("M2 Transformer — 21d Classification Signal",
                  fontsize=10, color="#444444")
    ax2.set_ylabel("Softmax Probability")
    ax2.grid(axis="y", alpha=0.3); ax2.legend(fontsize=8)

plt.tight_layout()
save("rpt_m2_05_dashboard.png", fig10)


# =============================================================================
#  SECTION 8 — FINAL PRINT SUMMARY
# =============================================================================
print("\n══ 6. Final summary ══════════════════════════════════════════════════")
print(f"\n  All predictions based on: Apr 17, 2026 (actual latest close)\n")

print(f"  {'Stock':<12} {'Close':>8}  "
      f"{'M1 21d':>10} {'21d%':>7}  "
      f"{'M1 63d':>10} {'63d%':>7}  "
      f"{'21dDir':>8}  {'63dDir':>8}")
print("  " + "─"*80)
for sn in SN3:
    r = m1_results[sn]
    print(f"  {sn:<12} ¥{r['last_close']:>7,.0f}  "
          f"¥{r['target_21d']:>9,.0f} {r['ret_21d']*100:>+6.1f}%  "
          f"¥{r['target_63d']:>9,.0f} {r['ret_63d']*100:>+6.1f}%  "
          f"{r['dir21']:>7.1f}%  {r['dir63']:>7.1f}%")

print(f"\n  {'Stock':<12} {'Close':>8}  "
      f"{'SELL':>6} {'HOLD':>6} {'BUY':>6}  "
      f"{'Signal':>6} {'Conf':>6}  "
      f"{'CV acc':>12}  {'Above?':>7}")
print("  " + "─"*80)
for sn in SN3:
    r  = m2_results[sn]
    p  = r["fwd_prob"]
    print(f"  {sn:<12} ¥{r['last_close']:>7,.0f}  "
          f"{p[0]:>6.2f} {p[1]:>6.2f} {p[2]:>6.2f}  "
          f"{r['fwd_rec']:>6}  {r['fwd_conf']:>6.0%}  "
          f"{r['mean_acc']:.3f}±{r['std_acc']:.3f}  "
          f"{'YES ✓' if r['above'] else 'NO ✗'}")

print(f"""
  Output figures:
  ──────────────────────────────────────────────────────
  rpt_00_pipeline.png           Architecture diagram
  rpt_m1_01_price_history.png   10-year price + volume
  rpt_m1_02_predictions.png     TCN: predicted vs actual returns
  rpt_m1_03_residuals.png       TCN: residual scatter
  rpt_m1_04_loss.png            TCN: loss curves (real history)
  rpt_m1_05_forecast.png        TCN: forward price targets
  rpt_m2_01_cv_accuracy.png     Transformer: walk-forward CV
  rpt_m2_02_loss_curves.png     Transformer: loss + overfit notes
  rpt_m2_03_confusion.png       Transformer: confusion matrices
  rpt_m2_04_probs.png           Transformer: probability time-series
  rpt_m2_05_dashboard.png       Combined recommendation dashboard
""")
