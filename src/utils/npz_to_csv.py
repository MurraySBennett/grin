import numpy as np
import pandas as pd
import os
import sys
from src.utils.config import *

def convert_npz_to_csv():
    """
    Loads data from the generated NumPy .npz file and saves it as a CSV.
    The CSV will contain the flattened confusion matrices, trial counts,
    model parameters, and model labels.
    """
    print(f"Checking for dataset file at: {DATASET_FILE}")
    if not os.path.exists(DATASET_FILE):
        print(f"Error: Dataset file not found at {DATASET_FILE}. Please run the data generation script first.")
        sys.exit(1)

    print("Loading data from .npz file...")
    data = np.load(DATASET_FILE)
    
    # Extract data from the NumPy arrays
    confusion_matrices = data['X']        # Shape: (N, 16) - flattened confusion matrices
    trial_counts = data['X_trials']       # Shape: (N, 4) - trial counts per stimulus
    parameters = data['y_params']         # Shape: (N, 26) - flattened model parameters
    model_labels = data['y_cls_label']    # Shape: (N,) - model names

    # Create a sequential sample ID for this dataset
    sample_ids = np.arange(len(model_labels))

    cm_columns = [f"cm_s{s+1}_r{r+1}" for s in range(4) for r in range(4)]
    trial_columns = [f"trials_s{s+1}" for s in range(4)]

    param_columns = (
        [f"mean_{i+1}" for i in range(8)] +
        [f"cov_mat_{s+1}{i+1}" for s in range(4) for i in range(4)] +
        [f"c_{i+1}" for i in range(2)]
    )
    
    data_dict = {
        'sample_id': sample_ids,
        **{col: confusion_matrices[:, i] for i, col in enumerate(cm_columns)},
        **{col: trial_counts[:, i] for i, col in enumerate(trial_columns)},
        **{col: parameters[:, i] for i, col in enumerate(param_columns)},
        "model_label": model_labels
    }
    
    df = pd.DataFrame(data_dict)
    
    print(f"Saving DataFrame to CSV file at: {DATASET_CSV_FILE}")
    df.to_csv(DATASET_CSV_FILE, index=False)
    
    print(f"Successfully converted .npz to CSV! File saved to {DATASET_CSV_FILE}")
    print("DataFrame head:")
    print(df.head())


if __name__ == "__main__":
    convert_npz_to_csv()
