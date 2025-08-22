# I have set this up to run AFTER the train_models script. A better programmer would make it independent, but here we are.
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


def create_test_dataset_csv():
    """
    Merges the main dataset CSV with the data splits log to create a 
    new CSV containing only the held-out test data.
    """
    # Define file paths
    output_file = os.path.join(SIMULATED_DATA_DIR, "grt_dataset_testsplit.csv")

    # Check that required files exist - it really should. This funciton runs in the same script after it's created..
    if not os.path.exists(DATASET_CSV_FILE):
        print(f"Error: Dataset file not found at {DATASET_CSV_FILE}.")
        sys.exit(1)
    if not os.path.exists(SIMULATED_DATA_SPLIT_LOG):
        print(f"Error: Data splits log file not found at {SIMULATED_DATA_SPLIT_LOG}. The split log file is saved when running: python -m scripts.train_models")
        sys.exit(1)
    print("Loading data from CSV files...")
    try:
        df_dataset = pd.read_csv(DATASET_CSV_FILE)
        df_splits = pd.read_csv(SIMULATED_DATA_SPLIT_LOG)
    except pd.errors.EmptyDataError:
        print("Error: One or both CSV files are empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Error: Failed to parse one or both CSV files. Check for formatting issues.")
        sys.exit(1)
    
    # Merge the two DataFrames on the 'sample_id' column then filter for 'test'
    print("Merging datasets on 'sample_id'...")
    merged_df = pd.merge(df_dataset, df_splits, on='sample_id', how='inner')
    test_split_df = merged_df[merged_df['data_split'] == 'test']
    test_split_df = test_split_df.drop(columns=['data_split', 'index'], errors='ignore')

    print(f"Saving test split data to: {output_file}")
    test_split_df.to_csv(output_file, index=False)
    

if __name__ == "__main__":
    convert_npz_to_csv()
    create_test_dataset_csv()
