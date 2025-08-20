import os
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- Directory Paths ---
# Directory to access models and utils
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MODELS_DIR = os.path.join(SRC_DIR, 'models')
UTILS_DIR = os.path.join(SRC_DIR, 'utils')

# Directory to save figures and plots.
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')
MODEL_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'models')

# Base data directory
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Directories for different data types
DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
SIMULATED_DATA_DIR = os.path.join(DATA_DIR, 'simulated_grt')

# The file path to save/load the simulated dataset.
DATASET_FILE = os.path.join(SIMULATED_DATA_DIR, 'grt_dataset.npz')

# Ensure all necessary directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(SIMULATED_DATA_DIR, exist_ok=True)


# --- Model Definitions & Curriculum ---
# The names of the model files to be trained.
MODEL_FILES = [
    'independent_models',
    'parallel_multi_task_model',
    'gated_multi_task_model',
    'cascaded_mc_dropout_model',
    'cascaded_weighted_uncertainty_model',
    'cascaded_non_bayesian_model',
]

# The curriculum stages, defining which models are added at each stage.
STAGED_CURRICULUM = [
    ['pi_ps_ds'],        # Stage 1
    ['rho1_ps_ds'],      # Stage 2
    ['pi_psa_ds', 'pi_psb_ds'],  # Stage 3
    ['rho1_psa_ds', 'rho1_psb_ds'], # Stage 4
    ['pi_ds', 'ps_ds'],  # Stage 5
    ['rho1_ds'],         # Stage 6
    ['psa_ds', 'psb_ds'],      # Stage 7
    ['ds'],              # Stage 8
]

# --- Data Generation Parameters ---
# The number of matrices to generate per model.
NUM_MATRICES_PER_MODEL = 25_000

# The range of trials per stimulus.
TRIALS_RANGE = (25, 1000)

# A small epsilon value to prevent division by zero in calculations.
EPSILON = tf.keras.backend.epsilon()

# --- Training Hyperparameters ---
TEST_SPLIT = 0.2
EPOCHS = 150
BATCH_SIZE = 128
ACTIVATION = 'tanh'  # 'tanh' or 'relu'
LEARNING_RATE = 1e-4
PATIENCE = 20
MIN_DELTA = 0.1

# Reduce learning rate on plateau parameters.
RLRP_FACTOR = 0.2
RLRP_PATIENCE = 10
RLRP_MIN_LR = 0.00001

NUM_MC_PREDICTIONS = 50
