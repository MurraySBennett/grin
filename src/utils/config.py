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
    'independent_models',
    'parallel_multi_task_model',
    'gated_multi_task_model',
    'cascaded_mc_dropout_model',
    'cascaded_weighted_uncertainty_model',
    'cascaded_non_bayesian_model',
    'customised_gate_control_model',
    'independent_separate_param_loss'
]

# The curriculum stages, defining which models are added at each stage.
STAGED_CURRICULUM = [
    ['pi_ps_ds', 'rho1_ps_ds'],
    ['pi_psa_ds', 'pi_psb_ds', 'rho1_psa_ds', 'rho1_psb_ds'],
    ['pi_ds', 'ps_ds', 'rho1_ds'],
    ['psa_ds', 'psb_ds', 'ds']
]

MODEL_NAMES = [
    'pi_ps_ds', 'rho1_ps_ds', 
    'pi_psa_ds', 'pi_psb_ds', 'rho1_psa_ds', 'rho1_psb_ds',
    'pi_ds', 'ps_ds', 'rho1_ds',
    'psa_ds', 'psb_ds', 'ds'
]

LSTM_MODEL_FILES = [
    'standard_lstm',
    'bidirectional_lstm',
    'gru_model',
    'cnn_lstm'
]

PARAM_NAMES = [
    'stim_0x', 'stim_0y', 'stim_1x', 'stim_1y', 'stim_2x', 'stim_2y', 'stim_3x', 'stim_3y', 
    'stim_0varx', 'stim_0cov', 'stim_0cov', 'stim0_vary', 
    'stim_1varx', 'stim_1cov', 'stim_1cov', 'stim1_vary', 
    'stim_2varx', 'stim_2cov', 'stim_2cov', 'stim2_vary', 
    'stim_3varx', 'stim_3cov', 'stim_3cov', 'stim3_vary', 
    'c1', 'c2'
]

# --- Data Generation Parameters ---
# The number of matrices to generate per model.
NUM_MATRICES_PER_MODEL = 25_000
NUM_PRETRAINING_MATRICES = 1_000

# The range of trials per stimulus.
TRIALS_RANGE = (1, 1000)

# A small epsilon value to prevent division by zero in calculations.
EPSILON = tf.keras.backend.epsilon()

DROPOUT = 0.1

# --- Training Hyperparameters ---
TEST_SPLIT = 0.2
EPOCHS = 150
BATCH_SIZE = 128
ACTIVATION = 'relu'  # 'tanh' or 'relu'
LEARNING_RATE = 0.001
PATIENCE = 30
MIN_DELTA = 0.05

PRE_EPOCHS = 100
PRE_PATIENCE = 40

# Reduce learning rate on plateau parameters.
RLRP_FACTOR = 0.2
RLRP_PATIENCE = 5
RLRP_MIN_LR = 0.00001

NUM_MC_PREDICTIONS = 50
