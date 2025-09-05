import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pandas as pd
import importlib.util
import json
import sys
import glob

from src.utils.config import *


def load_model_and_config(module_name):
    """
    Dynamically loads the model builder and config from a specified Python file.
    This function is now aligned with the dynamic import logic of the training script.
    """
    try:
        module_name = module_name.split(".")[0]
        model_module = importlib.import_module(f"src.models.{module_name}")
        model_builder_name = f"build_{module_name.split('_')[0].lower()}_models" if "independent" in module_name else f"build_{module_name.split('.')[0].lower()}_model"
        model_builder = getattr(model_module, model_builder_name)
        
        config_name = f"{module_name.split('.')[0].upper()}_CONFIG"
        config = getattr(model_module, config_name)
        return model_builder, config
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model from '{module_name}': {e}")
        return None, None

def load_data_and_splits(dataset_file, split_log_file):
    """
    Loads the dataset and the data split indices from the CSV log.
    Ensures that we use the exact same test set as the training script.
    """
    # Load the raw dataset
    try:
        data = np.load(dataset_file)
        X_combined = np.hstack([data['X'] / (np.repeat(data['X_trials'], 4, axis=1) + EPSILON), np.log(data['X_trials'] + EPSILON)])
        y_params_pruned = np.hstack([data['y_params'][:, 2:8], data['y_params'][:, 9:24:4], data['y_params'][:, 24:26]])
        y_cls_label = data['y_cls_label']
        X_trials_combined = data['X_trials']
    except FileNotFoundError:
        print(f"Dataset file '{dataset_file}' not found. Please run the training script first.")
        sys.exit(1)

    # Load the data split log
    try:
        split_log = pd.read_csv(split_log_file)
        test_indices = split_log[split_log['data_split'] == 'test']['sample_id'].values
    except FileNotFoundError:
        print(f"Data split log file '{split_log_file}' not found. Please ensure the training script ran successfully.")
        sys.exit(1)
    
    # Filter the data to get the held-out test set
    X_test = X_combined[test_indices]
    y_reg_test = y_params_pruned[test_indices]
    y_cls_label_test = y_cls_label[test_indices]
    X_trials_test = X_trials_combined[test_indices]

    return X_test, y_reg_test, y_cls_label_test, test_indices, X_trials_test

def predict_and_save(model, X_test, y_reg_test, y_cls_label_test, test_indices, X_trials_test, config, normalization_data):
    """
    Makes predictions with the loaded model and saves the results to a CSV file.
    This function now handles denormalization and various model types.
    """
    model_name = config['model_name']
    print(f"Predicting with model: {model_name}")

    y_pred_cls = None
    if config.get('is_multi_task', True):
        y_pred = model.predict(X_test, verbose=0)
        y_pred_cls = y_pred[0]
        y_pred_reg_normalized = y_pred[1]
    else:
        # Standard Independent Regression model: single output
        y_pred_reg_normalized = model.predict(X_test, verbose=0)

    # Denormalize the regression predictions before saving
    regression_mean = normalization_data['mean']
    regression_std = normalization_data['std']
    y_pred_reg = (y_pred_reg_normalized * regression_std) + regression_mean

    # Ensure predictions are 2D
    if y_pred_reg.ndim == 1:
        y_pred_reg = np.expand_dims(y_pred_reg, axis=1)

    # Create a DataFrame to save the results
    dfs = []
    # Loop over each parameter to create a long-form DataFrame
    for i in range(y_reg_test.shape[1]):
        df = pd.DataFrame({
            'sample_id': test_indices,
            'model_name': y_cls_label_test,
            'true_value': y_reg_test[:, i],
            'predicted_value': y_pred_reg[:, i],
            'parameter_name': PARAM_NAMES[i],
            'trial_count': X_trials_test[:, 0],
            'predicted_cls': np.argmax(y_pred_cls, axis=1) if y_pred_cls is not None else np.nan
        })
        dfs.append(df)
    
    # Concatenate all parameter DataFrames
    final_df = pd.concat(dfs)
    
    # Save to a CSV file
    output_path = os.path.join(MODEL_RESULTS_DIR, f"{model_name}_predictions.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Predictions for {model_name} saved to {output_path}")


def get_latest_model_path(model_name, is_multi_task=True, is_reg_model=False):
    """
    Finds the path of the final, best-performing model saved by ModelCheckpoint.
    """
    if is_multi_task:
        pattern = os.path.join(MODEL_RESULTS_DIR, f"{model_name}_*.h5")
        list_of_files = glob.glob(pattern)
        if not list_of_files:
            return None
        return min(list_of_files, key=os.path.getctime)
    else:
        suffix = "_reg_*.h5" if is_reg_model else "_cls_*.h5"
        pattern = os.path.join(MODEL_RESULTS_DIR, f"{model_name}{suffix}")
        list_of_files = glob.glob(pattern)
        if not list_of_files:
            return None
        return min(list_of_files, key=os.path.getctime)


if __name__ == '__main__':
    # Make sure the results directory exists
    os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)
    
    # Load the test data once at the start of the script
    X_test, y_reg_test, y_cls_label_test, test_indices, X_trials_test = load_data_and_splits(DATASET_FILE, SIMULATED_DATA_SPLIT_LOG)
    
    # Load the normalization parameters
    try:
        normalization_data = np.load(os.path.join(MODEL_RESULTS_DIR, 'regression_normalization.npz'))
    except FileNotFoundError:
        print("Normalization data file 'regression_normalization.npz' not found. Exiting.")
        sys.exit(1)

    for MODEL_FILE in MODEL_FILES:
        # Load the model configuration
        model_builder, config = load_model_and_config(MODEL_FILE)
        if not model_builder:
            continue
        model_name = config['model_name']
        
        try:
            if config.get('is_multi_task', True):
                model_path = get_latest_model_path(model_name, is_multi_task=True)
                if not model_path:
                    raise FileNotFoundError(f"No model found for {model_name}.")
                model = keras.models.load_model(model_path)
                predict_and_save(model, X_test, y_reg_test, y_cls_label_test, test_indices, X_trials_test, config, normalization_data)
            else:
                # Load the independent regression model
                reg_model_path = get_latest_model_path(model_name, is_multi_task=False, is_reg_model=True)
                if not reg_model_path:
                    raise FileNotFoundError(f"No regression model found for {model_name}.")
                
                # We need to rebuild the model to load the weights correctly
                _, reg_model = model_builder(X_test.shape[1], num_models=12)
                reg_model.load_weights(reg_model_path)
                predict_and_save(reg_model, X_test, y_reg_test, y_cls_label_test, test_indices, X_trials_test, config, normalization_data)

        except FileNotFoundError as e:
            print(f"Skipping prediction for {model_name}: {e}")
        except Exception as e:
            print(f"An error occurred while loading or predicting with {model_name}: {e}")
