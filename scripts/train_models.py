import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import skew
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import importlib.util
import json
import sys
import time

from src.utils.model_plotting_funcs import *
from src.utils.GRT_data_generator import GRTDataGenerator
from src.utils.config import *


def load_model_from_file(module_name):
    """Dynamically loads the model builder and config from a specified Python file."""
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

def log_data_splits_to_csv(data_full, idx_train_val, idx_test, filename=SIMULATED_DATA_SPLIT_LOG):
    """
    Creates and saves a CSV file logging the data split (train/validation/test) 
    for each sample in the dataset.
    """
    _, _, _, _, _, _, y_cls_name,_  = data_full
    split_labels = pd.Series(index=np.arange(len(y_cls_name)), dtype='object')
    split_labels.loc[idx_test] = 'test'
    split_labels.loc[idx_train_val] = 'train_val'
    data_log = pd.DataFrame({
        'sample_id': np.arange(len(y_cls_name)),
        'model_name': y_cls_name,
        'data_split': split_labels.values
    })
    data_log.to_csv(filename, index=False)
    print(f"Data split log saved to {filename}")
    
def normalise_var(train, test, mean, std):
    # Normalize the training and test regression targets
    train = (train - mean) / std
    test = (test - mean) / std
    return train, test
 
def denorm_var(preds, mean, std):
    return (preds * std) + mean


def compute_matrix_features(X, X_trials, epsilon=1e-8):
    n_samples = X.shape[0]
    n_classes = 4
    X_props = X / (np.repeat(X_trials, n_classes, axis=1) + epsilon)
    X_matrices = X_props.reshape(n_samples, n_classes, n_classes)

    features = []

    for mat, trials in zip(X_matrices, X_trials):
        # Flattened row/column statistics
        row_entropy = (-np.sum(mat * np.log(mat + epsilon), axis=1)).ravel()
        col_entropy = (-np.sum(mat * np.log(mat + epsilon), axis=0)).ravel()
        row_var = np.var(mat, axis=1).ravel()
        col_var = np.var(mat, axis=0).ravel()
        row_skew = np.array([skew(r) if np.std(r) > 1e-12 else 0.0 for r in mat]).ravel()

        diag_probs = np.diagonal(mat).ravel()
        off_diag_sum = (np.sum(mat, axis=1) - diag_probs).ravel()
        overall_accuracy = np.atleast_1d(np.sum(diag_probs) / (trials + epsilon))

        # Marginals
        x_marginal = np.atleast_1d(np.array([np.sum(mat[[0,2], :]), np.sum(mat[[1,3], :])]))
        y_marginal = np.atleast_1d(np.array([np.sum(mat[[0,1], :]), np.sum(mat[[2,3], :])]))

        # Bias/tendency
        row_sum = np.sum(mat, axis=1).ravel()
        col_sum = np.sum(mat, axis=0).ravel()
        row_bias = row_sum / (trials + epsilon)
        col_bias = col_sum / (trials + epsilon)

        # Confusability / discriminability
        confusability_off_diag = off_diag_sum / (row_sum + epsilon)
        discriminability = (diag_probs - np.mean(mat - np.diag(np.diag(mat)), axis=1)).ravel()

        # Concatenate everything safely
        feat_vector = np.hstack([
            mat.flatten(),
            row_entropy, col_entropy,
            row_var, col_var,
            row_skew,
            diag_probs, off_diag_sum,
            overall_accuracy,
            x_marginal, y_marginal,
            row_bias, col_bias,
            confusability_off_diag,
            discriminability
        ])

        features.append(feat_vector)

    return np.array(features)

def prepare_regression_targets(y_means, y_covs, y_crit, y_cls_ids):
    """Append class IDs to means and covs for regression inputs; leave crit unchanged."""
    class_ids = np.argmax(y_cls_ids, axis=1).reshape(-1, 1)
    y_means_w_id = np.concatenate([y_means, class_ids], axis=1)
    y_covs_w_id = np.concatenate([y_covs, class_ids], axis=1)
    y_crit_w_id = y_crit
    return y_means_w_id, y_covs_w_id, y_crit_w_id

if __name__ == '__main__':
    gen = GRTDataGenerator(num_matrices=NUM_MATRICES_PER_MODEL, trial_range=TRIALS_RANGE)

    if os.path.exists(DATASET_FILE):
        print("Loading pre-existing dataset...")
        data = np.load(DATASET_FILE)
        X = data['X']
        X_trials = data['X_trials']
        y_params = data['y_params']
        y_cls_id = data['y_model_cls']
        y_cls_name = data['y_cls_label']
        y_accuracy = np.sum(X[:, [0, 5, 10, 15]], axis=1) / np.sum(X_trials, axis=1)
    else:
        print(f"Generating a new dataset with {NUM_MATRICES_PER_MODEL} matrices per model...")
        X, y_params, X_trials, y_cls_id, y_cls_name = gen.generate_all_model_cms()
        y_accuracy = np.sum(X[:, [0, 5, 10, 15]], axis=1) / np.sum(X_trials, axis=1)
        np.savez(DATASET_FILE, X=X, X_trials=X_trials, y_params=y_params,
                 y_model_cls=y_cls_id, y_cls_label=y_cls_name)
        print("Dataset saved!")

    # --- Prepare regression and classification targets ---
    y_means = y_params[:, 2:8]
    y_covs = y_params[:, [9, 13, 17, 21]]
    y_crit = y_params[:, 24:26]
    y_cls_id = to_categorical(y_cls_id)

    # --- Feature extraction ---
    if os.path.exists(MATRIX_FEATURE_FILE):
        print(f"Loading {MATRIX_FEATURE_FILE}...")
        feature_data = np.load(MATRIX_FEATURE_FILE)
        X_input = feature_data['X_input']
    else:
        print("Generating matrix features...")
        X_input = compute_matrix_features(X, X_trials)
        np.savez(MATRIX_FEATURE_FILE, X_input=X_input)

    data_full = (X_input, X_trials, y_means, y_covs, y_crit, y_cls_id, y_cls_name, y_accuracy)

    # --- Split dataset ---
    train_val_idx, test_idx = train_test_split(
        np.arange(len(y_cls_name)),
        test_size=TEST_SPLIT,
        stratify=np.argmax(y_cls_id, axis=1),
        random_state=42
    )

    # --- Save split information (train+val vs test) ---
    log_data_splits_to_csv(data_full, train_val_idx, test_idx)

    # --- Index all arrays ---
    X_train_val_raw, X_test_raw = X_input[train_val_idx], X_input[test_idx]
    X_trials_train_val, X_trials_test = X_trials[train_val_idx], X_trials[test_idx]
    y_means_train_val_raw, y_means_test_raw = y_means[train_val_idx], y_means[test_idx]
    y_covs_train_val_raw, y_covs_test_raw = y_covs[train_val_idx], y_covs[test_idx]
    y_crit_train_val_raw, y_crit_test_raw = y_crit[train_val_idx], y_crit[test_idx]
    y_cls_train_val, y_cls_test = y_cls_id[train_val_idx], y_cls_id[test_idx]
    y_cls_name_train_val, y_cls_name_test = y_cls_name[train_val_idx], y_cls_name[test_idx]
    y_accuracy_train_val, y_accuracy_test = y_accuracy[train_val_idx], y_accuracy[test_idx]

    # --- Split train_val into train and validation ---
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train_val_raw)),
        test_size=0.25,
        stratify=np.argmax(y_cls_train_val, axis=1),
        random_state=42
    )

    X_train_raw, X_val_raw = X_train_val_raw[train_idx], X_train_val_raw[val_idx]
    y_means_train_raw, y_means_val_raw = y_means_train_val_raw[train_idx], y_means_train_val_raw[val_idx]
    y_covs_train_raw, y_covs_val_raw = y_covs_train_val_raw[train_idx], y_covs_train_val_raw[val_idx]
    y_crit_train_raw, y_crit_val_raw = y_crit_train_val_raw[train_idx], y_crit_train_val_raw[val_idx]
    y_cls_train, y_cls_val = y_cls_train_val[train_idx], y_cls_train_val[val_idx]
    y_cls_name_train, y_cls_name_val = y_cls_name_train_val[train_idx], y_cls_name_train_val[val_idx]
    y_accuracy_train, y_accuracy_val = y_accuracy_train_val[train_idx], y_accuracy_train_val[val_idx]

    # --- Normalize regression targets based only on TRAINING data ---
    means_mean = np.mean(y_means_train_raw, axis=0)
    means_std = np.std(y_means_train_raw, axis=0)
    covs_mean = np.mean(y_covs_train_raw, axis=0)
    covs_std = np.std(y_covs_train_raw, axis=0)
    crit_mean = np.mean(y_crit_train_raw, axis=0)
    crit_std = np.std(y_crit_train_raw, axis=0)

    # Save normalization params for denormalization later
    np.savez(os.path.join(MODEL_RESULTS_DIR, 'regression_normalization.npz'),
             means_mean=means_mean, means_std=means_std,
             covs_mean=covs_mean, covs_std=covs_std,
             crit_mean=crit_mean, crit_std=crit_std)

    # Normalize
    y_means_train, y_means_val, y_means_test = (
        (y_means_train_raw - means_mean) / means_std,
        (y_means_val_raw - means_mean) / means_std,
        (y_means_test_raw - means_mean) / means_std
    )
    y_covs_train, y_covs_val, y_covs_test = (
        (y_covs_train_raw - covs_mean) / covs_std,
        (y_covs_val_raw - covs_mean) / covs_std,
        (y_covs_test_raw - covs_mean) / covs_std
    )
    y_crit_train, y_crit_val, y_crit_test = (
        (y_crit_train_raw - crit_mean) / crit_std,
        (y_crit_val_raw - crit_mean) / crit_std,
        (y_crit_test_raw - crit_mean) / crit_std
    )

    # --- Normalize input features based only on TRAINING data ---
    X_mean = np.mean(X_train_raw, axis=0)
    X_std = np.std(X_train_raw, axis=0) + 1e-8

    X_train = (X_train_raw - X_mean) / X_std
    X_val = (X_val_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    # Save input normalization params
    np.savez(os.path.join(MODEL_RESULTS_DIR, 'input_feature_normalization.npz'),
             X_mean=X_mean, X_std=X_std)


    # --- Model name mapping ---
    model_name_to_idx = {name: i for i, name in enumerate(gen.model_names)}
    input_shape = X_input.shape[1]
    num_models = len(gen.model_names)

    # --- Training loop ---
    for MODEL_FILE in MODEL_FILES:
        model_builder, config = load_model_from_file(MODEL_FILE)
        if not model_builder:
            continue

        built_model = model_builder(input_shape, num_models, dropout_rate=DROPOUT, activation=ACTIVATION)
        is_multi_task = config.get('is_multi_task', False)

        if is_multi_task:
            model = built_model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss=config['losses'],
                loss_weights=config.get('loss_weights', None),
                metrics=config['metrics']
            )
        else:
            if isinstance(built_model, tuple):
                cls_model, reg_model = built_model
                cls_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=config['cls_losses'],
                    metrics=[config['cls_metrics']]
                )
                reg_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=config['reg_losses'],
                    loss_weights=config.get('loss_weights', None),
                    metrics=config['reg_metrics']
                )

        print(f"\n--- Training {config['model_name']} with Curriculum Learning ---")
        combined_history, cls_history, reg_history = {}, {}, {}
        cumulative_mask = np.zeros(len(X_train), dtype=bool)


        for stage_idx, (accuracy_range, stage_models) in enumerate(STAGED_CURRICULUM):
            print(f"\n--- Stage {stage_idx + 1}: Accuracy {accuracy_range}, Models {stage_models} ---")

            # Map model names to indices
            curr_indices = [model_name_to_idx[name] for name in stage_models]

            # Mask training samples belonging to the current stage
            stage_mask = np.isin(np.argmax(y_cls_train, axis=1), curr_indices)
            accuracy_mask = (y_accuracy_train >= accuracy_range[0]) & (y_accuracy_train < accuracy_range[1])
            cumulative_mask = np.logical_or(cumulative_mask, stage_mask & accuracy_mask)

            # Stage data (TRAINING only)
            X_stage = X_train[cumulative_mask]
            y_mean_stage = y_means_train[cumulative_mask]
            y_cov_stage = y_covs_train[cumulative_mask]
            y_crit_stage = y_crit_train[cumulative_mask]
            y_cls_stage = y_cls_train[cumulative_mask]

            # Split stage into train/val
            stage_indices = np.arange(len(X_stage))
            stage_train_idx, stage_val_idx = train_test_split(
                stage_indices,
                test_size=0.25,
                stratify=np.argmax(y_cls_stage, axis=1),
                random_state=42
            )

            X_train_stage, X_val_stage = X_stage[stage_train_idx], X_stage[stage_val_idx]
            y_mean_train_stage, y_mean_val_stage = y_mean_stage[stage_train_idx], y_mean_stage[stage_val_idx]
            y_cov_train_stage, y_cov_val_stage = y_cov_stage[stage_train_idx], y_cov_stage[stage_val_idx]
            y_crit_train_stage, y_crit_val_stage = y_crit_stage[stage_train_idx], y_crit_stage[stage_val_idx]
            y_cls_train_stage, y_cls_val_stage = y_cls_stage[stage_train_idx], y_cls_stage[stage_val_idx]

            # --- Prepare regression targets ---
            y_mean_train_stage_w_id, y_cov_train_stage_w_id, y_crit_train_stage_w_id = prepare_regression_targets(
                y_mean_train_stage, y_cov_train_stage, y_crit_train_stage, y_cls_train_stage
            )
            y_mean_val_stage_w_id, y_cov_val_stage_w_id, y_crit_val_stage_w_id = prepare_regression_targets(
                y_mean_val_stage, y_cov_val_stage, y_crit_val_stage, y_cls_val_stage
            )
            # --- Callbacks ---
            callbacks = [
                ReduceLROnPlateau(monitor='val_loss', factor=RLRP_FACTOR, patience=RLRP_PATIENCE, min_lr=RLRP_MIN_LR, verbose=1),
                EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, verbose=1, restore_best_weights=True)
            ]

            # --- Training ---
            if is_multi_task:
                train_targets = {'classification_output': y_cls_train_stage,
                                 'means_output': y_mean_train_stage_w_id,
                                 'cov_output': y_cov_train_stage_w_id,
                                 'crit_output': y_crit_train_stage_w_id}
                val_targets = {'classification_output': y_cls_val_stage,
                               'means_output': y_mean_val_stage_w_id,
                               'cov_output': y_cov_val_stage_w_id,
                               'crit_output': y_crit_val_stage_w_id}

                history = model.fit(
                    x=X_train_stage, y=train_targets,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_val_stage, val_targets),
                    callbacks=callbacks + [
                        ModelCheckpoint(
                            filepath=os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_{stage_idx+1}.h5"),
                            monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1
                        )
                    ],
                    verbose=2
                )
                for key, value in history.history.items():
                    combined_history.setdefault(key, []).extend(value)

            else:  # Independent model
                # Classification
                history_cls = cls_model.fit(
                    x=X_train_stage, y=y_cls_train_stage,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_val_stage, y_cls_val_stage),
                    callbacks=callbacks + [
                        ModelCheckpoint(
                            filepath=os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_cls_{stage_idx+1}.h5"),
                            monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1
                        )
                    ],
                    verbose=2
                )
                for key, value in history_cls.history.items():
                    cls_history.setdefault(key, []).extend(value)

                # Regression
                train_targets = {
                    'means_output': y_mean_train_stage_w_id,
                    'cov_output': y_cov_train_stage_w_id,
                    'crit_output': y_crit_train_stage_w_id
                }
                val_targets = {
                    'means_output': y_mean_val_stage_w_id,
                    'cov_output': y_cov_val_stage_w_id,
                    'crit_output': y_crit_val_stage_w_id
                }
                if config['model_name'] == "IndependentEnsemble":
                    reg_train_inputs = [X_train_stage, y_cls_train_stage]
                    reg_val_inputs = [X_val_stage, y_cls_val_stage]
                else:
                    reg_train_inputs = X_train_stage
                    reg_val_inputs = X_val_stage
                history_reg = reg_model.fit(
                    x=reg_train_inputs,
                    y=train_targets,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(reg_val_inputs, val_targets),
                    callbacks=callbacks + [
                        ModelCheckpoint(
                            filepath=os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_reg_{stage_idx+1}.h5"),
                            monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1
                        )
                    ],
                    verbose=2
                )
                for key, value in history_reg.history.items():
                    reg_history.setdefault(key, []).extend(value)

        # --- Final evaluation ---
        regression_params = np.load(os.path.join(MODEL_RESULTS_DIR, 'regression_normalization.npz'))
        
        # Denormalize predictions and compare to RAW test data
        if is_multi_task:
            final_combined_history = keras.callbacks.History()
            final_combined_history.history = combined_history
            plot_history(final_combined_history, FIGURES_DIR, config['model_name'])
            
            y_pred_cls, y_pred_means, y_pred_covs, y_pred_crit = model.predict(X_test)
            
            y_pred_means = denorm_var(y_pred_means, regression_params['means_mean'], regression_params['means_std'])
            y_pred_covs = denorm_var(y_pred_covs, regression_params['covs_mean'], regression_params['covs_std'])
            y_pred_crit = denorm_var(y_pred_crit, regression_params['crit_mean'], regression_params['crit_std'])

            final_mae_means = mean_absolute_error(y_means_test_raw, y_pred_means)
            final_mae_covs = mean_absolute_error(y_covs_test_raw, y_pred_covs)
            final_mae_crit = mean_absolute_error(y_crit_test_raw, y_pred_crit)

        else:  # Independent model
            # History plots (this part is correct)
            final_cls_history = keras.callbacks.History()
            final_cls_history.history = cls_history
            plot_history(final_cls_history, FIGURES_DIR, config['model_name'] + "_cls")
            
            final_reg_history = keras.callbacks.History()
            final_reg_history.history = reg_history
            plot_history(final_reg_history, FIGURES_DIR, config['model_name'] + "_reg")
            
            # Get predictions from the classification model
            y_pred_cls = cls_model.predict(X_test)
            
            if config['model_name'] == "IndependentEnsemble":
                predicted_class_ids = np.argmax(y_pred_cls, axis=1)
                y_pred_class_onehot = to_categorical(predicted_class_ids, num_classes=num_models)
                reg_test_inputs = [X_test, y_pred_class_onehot]
            else:
                reg_test_inputs = X_test
            y_pred_means, y_pred_covs, y_pred_crit = reg_model.predict(reg_test_inputs)

            # Denormalize predictions and compare to RAW test data
            y_pred_means = denorm_var(y_pred_means, regression_params['means_mean'], regression_params['means_std'])
            y_pred_covs = denorm_var(y_pred_covs, regression_params['covs_mean'], regression_params['covs_std'])
            y_pred_crit = denorm_var(y_pred_crit, regression_params['crit_mean'], regression_params['crit_std'])

            final_mae_means = mean_absolute_error(y_means_test_raw, y_pred_means)
            final_mae_covs = mean_absolute_error(y_covs_test_raw, y_pred_covs)
            final_mae_crit = mean_absolute_error(y_crit_test_raw, y_pred_crit)

        # Calculate final evaluation metrics
        final_accuracy = accuracy_score(np.argmax(y_cls_test, axis=1), np.argmax(y_pred_cls, axis=1))

        print(f"Final Test Accuracy: {final_accuracy:.4f}")
        print("=== Final Test MAE ===")
        print(f"Means: {final_mae_means:.3f}\nCovariances: {final_mae_covs:.3f}\nCrits: {final_mae_crit:.3f}")
            
        # Plotting
        plot_confusion_matrix(y_cls_test, y_pred_cls, gen.model_names, FIGURES_DIR, config['model_name'] + "_cls")

        plot_regression_performance(
            np.hstack([y_means_test, y_covs_test, y_crit_test]),
            np.hstack([y_pred_means, y_pred_covs, y_pred_crit]),
            FIGURES_DIR, config['model_name'] + "_reg"
        )
