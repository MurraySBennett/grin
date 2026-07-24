"""
config.py — project-wide paths, constants, and hyperparameters for GRIN.

Import as `from src.config import ...` (works from the project root after
`pip install -e .`, or with PYTHONPATH=.). Model classes and parameter names are
re-exported from grt_model, the single source of truth. This module has no heavy
dependencies (no torch, no TF), so lightweight code can import it freely.
"""
import os
from src.grt_model import (MODEL_NAMES, MODEL_SPECS, PARAM_NAMES, DATA_DF, n_free_params)

N_PARAMS = len(PARAM_NAMES)          # 12 identified parameters

# --- Paths (absolute, cwd-independent) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR  = os.path.join(PROJECT_ROOT, "src")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
DATA_DIR           = os.path.join(PROJECT_ROOT, "data")
SIMULATED_DATA_DIR = os.path.join(DATA_DIR, "simulated")
REAL_DATA_DIR      = os.path.join(DATA_DIR, "real")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR   = os.path.join(RESULTS_DIR, "models")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")
MLE_FITS_DIR = os.path.join(RESULTS_DIR, "mle_fits")
DATASET_FILE     = os.path.join(SIMULATED_DATA_DIR, "grt_dataset.npz")
DATASET_CSV_FILE = os.path.join(SIMULATED_DATA_DIR, "grt_dataset.csv")
COVERAGE_FIGURE  = os.path.join(FIGURES_DIR, "coverage_report.png")
MODEL_FILE       = os.path.join(MODELS_DIR, "npe_model.pt")
TRAINING_HISTORY_DIR = os.path.join(RESULTS_DIR, "training_history")
for _d in (SIMULATED_DATA_DIR, REAL_DATA_DIR, MODELS_DIR, FIGURES_DIR, MLE_FITS_DIR, TRAINING_HISTORY_DIR):
    os.makedirs(_d, exist_ok=True)

# --- Data generation ---
# TRIAL_RANGE: set this to BRACKET what your real experiments collect. The network is
# calibrated for the range it trains on. The low end matters for adaptive/real-time use.
N_PER_CLASS = 100_000          # matrices per model class (x12). Generation is cheap.
TRIAL_RANGE = (1, 1000)       # per-participant BASE trial count, sampled log-uniformly
# TRIAL_IMBALANCE: how unbalanced the 4 per-stimulus trial counts within one matrix may be.
# A single base count is drawn per participant (over TRIAL_RANGE); each stimulus then keeps a
# fraction in [1 - TRIAL_IMBALANCE, 1] of it, modelling attrition (lapses, misses, cleaning).
# So imbalance is bounded and proportional to set size — 0.35 => the smallest stimulus keeps
# >=65% of the largest. 0.0 == perfectly balanced. Raise it to expose the net to messier data.
TRIAL_IMBALANCE = 0.35
Z_MAX = 3.0                   # max |z| (d'~3 = near-ceiling discrimination, ~93% correct)
R_MAX = 0.9                   # max |perceptual correlation| (beyond this is rare + near-singular)
DATA_SEED = 42

# --- RT pipeline (optional; only if your data include response times) ---
# The RT generator is IDENTICAL to the counts-only generator in every respect — same prior,
# same Z_MAX/R_MAX, same TRIAL_RANGE, same TRIAL_IMBALANCE, same N_PER_CLASS, same model classes — except that it
# ALSO emits the response times those same trials produced. Counts and RTs are matched by
# construction (one perceptual sample per trial determines both the response and the RT).
# It is vectorised, so it is fast enough to use the same N as the counts pipeline.
RT_DRIFT_SD = 0.35            # LBA drift-rate noise
# The RT model has a much heavier task (100 input features; 5 output heads: GRT params,
# 3 constructs, 5-way architecture, 4 LBA params), so it gets more capacity by default.
# Tune with: python validation/sweeps.py --only capacity  (watch CALIBRATION, not just MAE —
# bigger networks can become overconfident, and calibration is what GRIN's value rests on).
RT_HIDDEN_LAYERS = (192, 192, 192)
RT_DROPOUT = 0.0
RT_DATASET_FILE = os.path.join(SIMULATED_DATA_DIR, "grt_rt_dataset.npz")
RT_MODEL_FILE = os.path.join(MODELS_DIR, "npe_rt_model.pt")

# --- Model / inference (Phase 2, PyTorch) ---
N_INPUT = 16 + 4                 # row proportions (16) + log10 trial counts (4)
NPE_HEAD = "gaussian_full"       # "gaussian_full" | "gaussian_diag" | "flow" (future: zuko)
HIDDEN_LAYERS = (128, 128, 128)      # counts-only model (20 input features)
ACTIVATION = "tanh"
DROPOUT = 0.1
DEVICE = "cuda"                  # train/predict fall back to "cpu" if unavailable

# --- Training hyperparameters ---
VAL_SPLIT = 0.2
EPOCHS = 200
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
PATIENCE = 20
MIN_DELTA = 1e-3
RLRP_FACTOR = 0.2
RLRP_PATIENCE = 5
RLRP_MIN_LR = 1e-5
TRAIN_SEED = 0

# Small constant to avoid division by zero.
EPSILON = 1e-7
