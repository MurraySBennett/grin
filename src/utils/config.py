import os
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- Directory Paths ---
# Directory to access models and utils
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MODELS_DIR = os.path.join(SRC_DIR, 'models')
PRETRAINED_MODELS_DIR = os.path.join(MODELS_DIR, 'pretrained')
UTILS_DIR = os.path.join(SRC_DIR, 'utils')

# Directory to save figures and plots.
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
PRETRAINED_FIGURES_DIR = os.path.join(FIGURES_DIR, 'pretraining')
MODEL_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'models')
PRETRAINED_MODEL_RESULTS_DIR = os.path.join(MODEL_RESULTS_DIR, 'pretrained')

# Base data directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Directories for different data types
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SIMULATED_DATA_DIR = os.path.join(DATA_DIR, 'simulated_grt')
SIMULATED_DATA_SPLIT_LOG = os.path.join(SIMULATED_DATA_DIR, "data_splits_log.csv")

# The file path to save/load the simulated dataset.
DATASET_FILE = os.path.join(SIMULATED_DATA_DIR, 'grt_dataset.npz')
DATASET_CSV_FILE = os.path.join(SIMULATED_DATA_DIR, 'grt_dataset.csv')
TRIAL_BY_TRIAL_FIAL = os.path.join(SIMULATED_DATA_DIR, 'trial_by_trial_dataset.npz')
MATRIX_FEATURE_FILE = os.path.join(SIMULATED_DATA_DIR, 'matrix_features.npz')

# Ensure all necessary directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PRETRAINED_FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)
os.makedirs(PRETRAINED_MODEL_RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(SIMULATED_DATA_DIR, exist_ok=True)


# --- Model Definitions & Curriculum ---
# The names of the model files to be trained.
MODEL_FILES = [
    'independent.py', 
    'independent_ensemble.py',
    'parallel.py',
    'regressor_classifier_cascade.py',
    'classifier_regressor_cascade.py',
    'shared_backbone.py',
    'shared_specialised.py',
    'gated_heads.py',
    'attention.py',
    'progressive_specialisation.py'
]

MODEL_NAMES = [
    'pi_ps_ds', 'rho1_ps_ds', 
    'pi_psa_ds', 'pi_psb_ds', 'rho1_psa_ds', 'rho1_psb_ds',
    'pi_ds', 'ps_ds', 'rho1_ds',
    'psa_ds', 'psb_ds', 'ds'
]
# STAGED_CURRICULUM = [
#     ((0.6, 0.8), ['pi_ps_ds', 'rho1_ps_ds', 'ps_ds', 'pi_ds', 'rho1_ds']), # perceptual independence
#     ((0.6, 0.8), ['pi_psa_ds', 'pi_psb_ds', 'rho1_psa_ds', 'rho1_psb_ds', 'psa_ds', 'psb_ds', 'ds']), # perceptual separability
#     ((0.0, 1.0), [
#         'pi_ps_ds', 'rho1_ps_ds', 'pi_psa_ds', 'pi_psb_ds', 'rho1_psa_ds', 'rho1_psb_ds',
#         'pi_ds', 'ps_ds', 'rho1_ds', 'psa_ds', 'psb_ds', 'ds'
#         ])
# ]


STAGED_CURRICULUM = [
    ((0.0, 1.0), MODEL_NAMES)
]

LSTM_MODEL_FILES = [
    'standard_lstm',
    'bidirectional_lstm',
    'gru_model',
    'cnn_lstm'
]

PARAM_NAMES = [
    'stim_1x', 'stim_1y', 'stim_2x', 'stim_2y', 'stim_3x', 'stim_3y', 
    'stim_0cov', 'stim_1cov', 'stim_2cov', 'stim_3cov',
    'c1', 'c2'
]

# --- Data Generation Parameters ---
# The number of matrices to generate per model.
# your current version is terrible and is trying to generate n_matrices_per_bin
NUM_PRETRAINING_MATRICES = 1_000
MIN_MATRIX_ACCURACY = 25 #you could set this to 0 to include non-random errors, but probably better as an empirical question (e.g., model performance w and w/o the below-chance data included).
MAX_MATRIX_ACCURACY = 100
MATRIX_ACCURACY_BIN_WIDTH = 5#2.5
MATRIX_ACCURACY_BINS = int((MAX_MATRIX_ACCURACY - MIN_MATRIX_ACCURACY) // MATRIX_ACCURACY_BIN_WIDTH) + 1

NUM_MATRICES_PER_ACCURACY_BIN = 5 # 500
NUM_MATRICES_PER_MODEL = MATRIX_ACCURACY_BINS * NUM_MATRICES_PER_ACCURACY_BIN 

# The range of trials per stimulus.
TRIALS_RANGE = (1, 1000)

USE_PRETRAINED_MODELS = False
# A small epsilon value to prevent division by zero in calculations.
EPSILON = tf.keras.backend.epsilon()

DROPOUT = 0.2

# --- Training Hyperparameters ---
TEST_SPLIT = 0.2
EPOCHS = 2#200
BATCH_SIZE = 256
ACTIVATION = 'tanh'  # 'tanh' or 'relu'
LEARNING_RATE = 0.001
PATIENCE = 20
MIN_DELTA = 0.01

PRE_EPOCHS = 100
PRE_PATIENCE = 40

# Reduce learning rate on plateau parameters.
RLRP_FACTOR = 0.2
RLRP_PATIENCE = 5
RLRP_MIN_LR = 0.00001

NUM_MC_PREDICTIONS = 50

