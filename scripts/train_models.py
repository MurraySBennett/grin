import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import importlib.util
import json
import sys
import time

# These imports are assumed to be available from your project structure
from src.utils.model_plotting_funcs import *
from src.utils.GRT_data_generator import GRTDataGenerator
from src.models.parallel_multi_task_model import get_parallel_multi_task_model_config
from src.models.cascaded_non_bayesian_model import get_cascaded_non_bayesian_model_config
from src.models.cascaded_mc_dropout_model import get_cascaded_mc_dropout_model_config
from src.models.cascaded_weighted_uncertainty_model import get_cascaded_weighted_uncertainty_model_config
from src.models.gated_multi_task_model import get_gated_multi_task_model_config
from src.models.customised_gate_control_model import get_customised_gate_control_model_config
from src.models.independent_separate_param_loss import get_independent_separate_param_loss_config, make_regression_losses, convert_targets_to_scales_corr, convert_scales_corr_to_cov

from src.utils.config import *

# Define whether to use pre-trained models. Set this to False to train from scratch.
USE_PRETRAINED_MODELS = True

# The denormalization function is no longer needed since we are using raw values.

def load_model_from_file(module_name):
    """Dynamically loads the model builder and config from a specified Python file."""
    try:
        model_module = importlib.import_module(f"src.models.{module_name}")
        return getattr(model_module, f"get_{module_name.lower()}_config")()
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model from '{module_name}': {e}")
        return None, None


def log_data_splits_to_csv(all_data, train_val_indices, test_indices, filename="data_splits_log.csv"):
    """
    Creates and saves a CSV file logging the data split (train/validation/test) 
    for each sample in the dataset.
    
    Args:
        all_data (tuple): The tuple containing all dataset arrays before splitting.
        train_val_indices (np.array): The indices of samples assigned to the
                                      combined training and validation set.
        test_indices (np.array): The indices of samples assigned to the test set.
        filename (str): The name of the output CSV file.
    """
    # Unpack the all_data tuple to get the necessary arrays
    X_combined, X_trials, y_params, y_model_cls, y_cls_label = all_data
    split_labels = pd.Series(index=np.arange(len(y_cls_label)), dtype='object')
    split_labels.loc[test_indices] = 'test'
    split_labels.loc[train_val_indices] = 'train_val' # This will be split further later - but distinction is unnecessary for our analysis. I think..

    # Create the DataFrame
    data_log = pd.DataFrame({
        'sample_id': np.arange(len(y_cls_label)),
        'model_name': y_cls_label,
        'data_split': split_labels.values
    })
    
    # Save the DataFrame to a CSV file
    data_log.to_csv(filename, index=False)
    print(f"Data split log saved to {filename}")
    
 
def train_and_evaluate_model(model, X_train, train_targets, X_val, val_targets, config):
    """Trains and evaluates a single Keras model, capturing timing and model size."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
        ModelCheckpoint(filepath=os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_best_model.h5"),
                        monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # --- Start Timer ---
    start_time = time.time()
    
    history = model.fit(
        X_train,
        train_targets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, val_targets),
        callbacks=callbacks
    )

    # --- Stop Timer ---
    end_time = time.time()
    training_time_sec = end_time - start_time
    
    # Save history
    history_path = os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_history.json")
    serializable_history = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f)
    print(f"Training history saved to: {history_path}") 
    
    # Save computational metrics
    metrics_path = os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}_computational_metrics.json")
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    computational_metrics = {
        'training_time_seconds': training_time_sec,
        'trainable_parameters': int(trainable_params),
        'best_epoch': int(np.argmin(history.history['val_loss'])) + 1
    }
    with open(metrics_path, 'w') as f:
        json.dump(computational_metrics, f)
    print(f"Computational metrics saved to: {metrics_path}")

    return model, history


if __name__ == '__main__':
    gen = GRTDataGenerator(num_matrices=NUM_MATRICES_PER_MODEL, trial_range=TRIALS_RANGE)

    if os.path.exists(DATASET_FILE):
        print("Loading pre-existing dataset...")
        data = np.load(DATASET_FILE)
        X = data['X']
        X_trials = data['X_trials']
        y_params = data['y_params']
        y_model_cls = data['y_model_cls']
        y_cls_label = data['y_cls_label']

    else:
        print(f"Generating a new dataset with {NUM_MATRICES_PER_MODEL} matrices per model...")
        X, y_params, X_trials, y_model_cls, y_cls_label = gen.generate_all_model_cms()
        np.savez(DATASET_FILE, X=X, X_trials=X_trials, y_params=y_params, y_model_cls=y_model_cls, y_cls_label=y_cls_label)
        print("Dataset saved!")
    
    model_names = gen.model_names
    y_model_cls = to_categorical(y_model_cls)

    X_proportions = X / (np.repeat(X_trials, 4, axis=1) + EPSILON)
    X_trials_log = np.log(X_trials + EPSILON)
    X_combined = np.hstack([X_proportions, X_trials_log])

    all_data = (X_combined, X_trials, y_params, y_model_cls, y_cls_label)

    original_indices = np.arange(len(y_cls_label))
    train_val_indices, test_indices = train_test_split(
        original_indices,
        test_size=TEST_SPLIT, 
        stratify=y_model_cls, 
        random_state=42
    )
    log_data_splits_to_csv(all_data, train_val_indices, test_indices, filename=os.path.join(SIMULATED_DATA_DIR, "data_splits_log.csv"))

    # 1. First split: Create a held-out test set (20%)
    (X_train_val, X_test, X_trials_train_val, X_trials_test, 
     y_reg_train_val, y_reg_test, y_cls_train_val, y_cls_test, 
     y_cls_label_train_val, y_cls_label_test) = train_test_split(
         *all_data, test_size=TEST_SPLIT, stratify=y_model_cls, random_state=42
    )

    # Get the mapping from model name to numeric index for filtering
    model_name_to_idx = {name: i for i, name in enumerate(model_names)}

    # Define these variables BEFORE the loop
    input_shape = X_combined.shape[1]
    num_models = len(model_names)
    num_params = y_params.shape[1]
    
    # Loop through each model architecture to train it with the curriculum
    for MODEL_FILE in MODEL_FILES:
        model_builder, config = load_model_from_file(MODEL_FILE)
        if not model_builder:
            continue
        
        print(f"\n--- Training {config['model_name']} with Curriculum Learning ---")
        if MODEL_FILE == "independent_separate_param_loss":
            _, reg_model = model_builder(input_shape, num_models, num_params, dropout_rate=DROPOUT, activation=ACTIVATION)
            if USE_PRETRAINED_MODELS:
                pretrained_reg_path = os.path.join(PRETRAINED_MODELS_DIR, f"{config['model_name']}_pretrained.h5")
                if os.path.exists(pretrained_reg_path):
                    print(f"Loading pre-trained regression weights for {config['model_name']}...")
                    reg_model.load_weights(pretrained_reg_path)
                else:
                    print(f"No pre-trained weights found for {config['model_name']}. Training from scratch.")
            model_to_train = reg_model
                
        # Initialize the model(s) once before the curriculum loop
        if config.get('is_multi_task', True):
            model = model_builder(input_shape, num_models, num_params, dropout_rate=DROPOUT, activation=ACTIVATION)
            if USE_PRETRAINED_MODELS:
                pretrained_path = os.path.join(PRETRAINED_MODELS_DIR, f"{config['model_name']}_pretrained.h5")
                if os.path.exists(pretrained_path):
                    print(f"Loading pre-trained weights for {config['model_name']}...")
                    try:
                        # by_name=True: Matches layers by name instead of by order.
                        # skip_mismatch=True: Skips layers that have a shape mismatch.
                        model.load_weights(pretrained_path, by_name=True, skip_mismatch=True)
                        print(f"Successfully loaded shared layer weights from '{pretrained_path}'.")
                    except Exception as e:
                        print(f"Warning: Could not load pre-trained weights. Error: {e}")
                        print("The model will be trained from scratch. Check the model architecture and file path.")
                else:
                    print(f"Warning: Pre-trained weights not found at {pretrained_path}. Training from scratch.")

        else: # Independent models
            if MODEL_FILE == "independent_separate_param_loss":  # This is GPT-independent
                _, reg_model = model_builder(input_shape, num_models, num_params, dropout_rate=DROPOUT, activation=ACTIVATION)
                if USE_PRETRAINED_MODELS:
                    pretrained_reg_path = os.path.join(PRETRAINED_MODELS_DIR, f"{config['model_name']}_pretrained.h5")
                    if os.path.exists(pretrained_reg_path):
                        print(f"Loading pre-trained regression weights for {config['model_name']}...")
                        reg_model.load_weights(pretrained_reg_path)
                    else:
                        print(f"No pre-trained weights found for {config['model_name']}. Training from scratch.")
                model_to_train = reg_model
            else:
                cls_model, reg_model = model_builder(input_shape, num_models, num_params, dropout_rate=DROPOUT, activation=ACTIVATION)
                if USE_PRETRAINED_MODELS:
                    pretrained_reg_path = os.path.join(PRETRAINED_MODELS_DIR, f"{config['model_name']}_pretrained.h5")
                    if os.path.exists(pretrained_reg_path):
                        print(f"Loading pre-trained regression weights for {config['model_name']}...")
                        reg_model.load_weights(pretrained_reg_path)
                    else:
                        print(f"Warning: Pre-trained regression weights not found at {pretrained_reg_path}. Training from scratch.")

        current_curriculum_labels = []


        # Curriculum Training Loop
        for stage_idx, stage_models in enumerate(STAGED_CURRICULUM):
            print(f"\n--- Stage {stage_idx + 1}: Adding models {stage_models} ---")
            current_curriculum_labels.extend(stage_models)
            
            # Get the numeric indices for the current curriculum models
            current_curriculum_indices = [model_name_to_idx[name] for name in current_curriculum_labels]
            
            # Filter the train_val dataset based on the current curriculum
            mask = np.isin(np.argmax(y_cls_train_val, axis=1), current_curriculum_indices)
            X_train_curriculum_val = X_train_val[mask]
            y_reg_train_val_curriculum = y_reg_train_val[mask]
            y_cls_train_val_curriculum = y_cls_train_val[mask]
            
            # Re-split the filtered data for this stage into train and val sets
            (X_train_stage, X_val_stage, 
             y_reg_train_stage, y_reg_val_stage, 
             y_cls_train_stage, y_cls_val_stage) = train_test_split(
                 X_train_curriculum_val, y_reg_train_val_curriculum, y_cls_train_val_curriculum,
                 test_size=0.25, stratify=y_cls_train_val_curriculum, random_state=42
            )

            print(f"Number of training samples in this stage: {X_train_stage.shape[0]}")
            print(f"Number of validation samples in this stage: {X_val_stage.shape[0]}")
            print(f"Batches per epoch: {X_train_stage.shape[0] / BATCH_SIZE}") 
            if MODEL_FILE == "independent_separate_param_loss":
                y_reg_train = convert_targets_to_scales_corr(y_reg_train_stage)
                y_reg_val = convert_targets_to_scales_corr(y_reg_val_stage)
            else:
                y_reg_train = y_reg_train_stage
                y_reg_val = y_reg_val_stage

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
            ]
            
            # Use the final filename for the ModelCheckpoint in the last stage
            is_final_stage = (stage_idx == len(STAGED_CURRICULUM) - 1)
            final_filepath = os.path.join(MODEL_RESULTS_DIR, f"{config['model_name']}.h5")

            if config.get('is_multi_task', True):
                model_name = config['model_name']
                # Determine the filename for this stage
                filepath = final_filepath if is_final_stage else os.path.join(MODEL_RESULTS_DIR, f"{model_name}_stage_{stage_idx}_best_model.h5")
                callbacks.append(ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, verbose=1))
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=config['losses'],
                    loss_weights=config.get('loss_weights', None),
                    metrics=config['metrics']
                )

                train_targets = {'classification_output': y_cls_train_stage, config['output_names'][1]: y_reg_train}
                val_targets = {'classification_output': y_cls_val_stage, config['output_names'][1]: y_reg_val}

                model.fit(
                    x=X_train_stage, y=train_targets,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_val_stage, val_targets),
                    callbacks=callbacks, verbose=1
                )
                
                # Load the best weights from the last stage's ModelCheckpoint
                model.load_weights(filepath)
                
                
            else:  # Independent models
                if MODEL_FILE == "independent_separate_param_loss":
                    reg_model_name = f"{config['model_name']}_Regression"
                    reg_filepath = os.path.join(MODEL_RESULTS_DIR, f"{reg_model_name}.h5") if is_final_stage else os.path.join(MODEL_RESULTS_DIR, f"{reg_model_name}_stage_{stage_idx}_best_model.h5")

                    reg_callbacks = [
                        EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
                        ModelCheckpoint(filepath=reg_filepath, monitor='val_loss', save_best_only=True, verbose=1)
                    ]

                    # Compile & train regression model
                    reg_model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                        loss=config['losses'],
                        loss_weights=config['loss_weights'],
                        metrics=config['metrics']
                    )
                    reg_model.fit(
                        X_train_stage,
                        {
                            'means_out': y_reg_train[:, 0:8],
                            'scales_corr_out': y_reg_train[:, 8:20],
                            'crit_out': y_reg_train[:, 20:22]
                        },
                        validation_data=(
                            X_val_stage,
                            {
                                'means_out': y_reg_val[:, 0:8],
                                'scales_corr_out': y_reg_val[:, 8:20],
                                'crit_out': y_reg_val[:, 20:22]
                            }
                        ),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=reg_callbacks,
                        verbose=1
                    )
                else:
                    cls_model_name = f"{config['model_name']}_Classification"
                    reg_model_name = f"{config['model_name']}_Regression"
                    
                    # Determine the filename for this stage
                    cls_filepath = os.path.join(MODEL_RESULTS_DIR, f"{cls_model_name}.h5") if is_final_stage else os.path.join(MODEL_RESULTS_DIR, f"{cls_model_name}_stage_{stage_idx}_best_model.h5")
                    reg_filepath = os.path.join(MODEL_RESULTS_DIR, f"{reg_model_name}.h5") if is_final_stage else os.path.join(MODEL_RESULTS_DIR, f"{reg_model_name}_stage_{stage_idx}_best_model.h5")

                    # Define callbacks for both models
                    cls_callbacks = [
                        EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
                        ModelCheckpoint(filepath=cls_filepath, monitor='val_loss', save_best_only=True, verbose=1)
                    ]
                    
                    reg_callbacks = [
                        EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
                        ModelCheckpoint(filepath=reg_filepath, monitor='val_loss', save_best_only=True, verbose=1)
                    ]
                    
                    # Train the classification model
                    print("\n--- Training Independent Classification Model ---")
                    cls_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=config['losses']['classification_output'], metrics=[config['metrics']['classification_output']])
                    cls_model.fit(x=X_train_stage, y=y_cls_train_stage, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val_stage, y_cls_val_stage), callbacks=cls_callbacks, verbose=1)
                    cls_model.load_weights(cls_filepath)

                    # Train the regression model
                    print("\n--- Training Independent Regression Model ---")

 
                    reg_model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=config['losses']['regression_output'], metrics=[config['metrics']['regression_output']])
                    reg_model.fit(x=X_train_stage, y=y_reg_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val_stage, y_reg_val), callbacks=reg_callbacks, verbose=1)
                reg_model.load_weights(reg_filepath)
                
        # --- Final Evaluation on the held-out test set ---
        print("\n--- Final Evaluation on Full Test Set ---")

        # Determine predictions based on model type
        if config.get('is_multi_task', True):
            # Multi-task model: y_pred is a list [cls_output, reg_output]
            y_pred_list = model.predict(X_test)
            y_pred_cls = y_pred_list[0]
            y_pred_reg = y_pred_list[1]

        elif MODEL_FILE == "independent_separate_param_loss":
            # GPT-independent model: may return dict or list
            y_pred_dict = reg_model.predict(X_test)
            y_pred_reg = convert_scales_corr_to_cov(
                np.hstack([y_pred_dict['means_out'], y_pred_dict['scales_corr_out'], y_pred_dict['crit_out']])
            )
            y_pred_reg = convert_scales_corr_to_cov(np.hstack([y_means_pred, y_scales_pred, y_crit_pred]))
            y_pred_cls = np.zeros_like(y_cls_test)

        else:
            # Standard independent model
            y_pred_cls = cls_model.predict(X_test)
            y_pred_reg = reg_model.predict(X_test)

        # Ensure predictions are 2D
        if y_pred_reg.ndim == 1:
            y_pred_reg = np.expand_dims(y_pred_reg, axis=1)

        # Compute metrics
        final_accuracy = accuracy_score(np.argmax(y_cls_test, axis=1), np.argmax(y_pred_cls, axis=1))
        final_mae = mean_absolute_error(y_reg_test, y_pred_reg)

        print(f"Final Test Accuracy: {final_accuracy:.4f}")
        print(f"Final Test MAE: {final_mae:.4f}")

        # Plot confusion matrix for classification
        plot_confusion_matrix(
            y_cls_test, y_pred_cls, model_names, FIGURES_DIR, 
            f"{config['model_name']}_Classification"
        )

        # Plot regression performance
        num_outputs = y_pred_reg.shape[1]
        num_cols = 4
        num_rows = int(np.ceil(num_outputs / num_cols))
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), squeeze=False)
        fig.suptitle(f'True vs. Predicted Values for {config["model_name"]}', fontsize=16)

        for i in range(num_outputs):
            ax = axs.flatten()[i]
            ax.scatter(y_reg_test[:, i], y_pred_reg[:, i], alpha=0.5)
            line_min = min(y_reg_test[:, i].min(), y_pred_reg[:, i].min())
            line_max = max(y_reg_test[:, i].max(), y_pred_reg[:, i].max())
            ax.plot([line_min, line_max], [line_min, line_max], 'r--')
            ax.text(0.05, 0.95, f'{PARAM_NAMES[i]}', transform=ax.transAxes, fontsize=12, verticalalignment='top')
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        regression_plot_path = os.path.join(FIGURES_DIR, f"{config['model_name']}_regression_performance.png")
        plt.savefig(regression_plot_path)
        plt.show()

        # Print summary statistics for regression
        print_summary_statistics(y_reg_test, y_pred_reg, config['model_name'], PARAM_NAMES)
