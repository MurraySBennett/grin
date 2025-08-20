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

from src.utils.model_plotting_funcs import *
from src.utils.GRT_data_generator import GRTDataGenerator
from src.models.parallel_multi_task_model import get_parallel_multi_task_model_config
from src.models.cascaded_non_bayesian_model import get_cascaded_non_bayesian_model_config
from src.models.cascaded_mc_dropout_model import get_cascaded_mc_dropout_model_config
from src.models.cascaded_weighted_uncertainty_model import get_cascaded_weighted_uncertainty_model_config
from src.models.gated_multi_task_model import get_gated_multi_task_model_config
from src.models.independent_models import get_independent_models_config

from src.utils.config import *

def denormalize_predictions(y_pred_normalized, min_vals, max_vals):
    """Denormalizes predicted values back to their original scale."""
    return y_pred_normalized * (max_vals - min_vals) + min_vals

def load_model_from_file(module_name):
    """Dynamically loads the model builder and config from a specified Python file."""
    try:
        model_module = importlib.import_module(f"src.models.{module_name}")
        return getattr(model_module, f"get_{module_name.lower()}_config")()
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model from '{module_name}': {e}")
        return None, None

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

    # 1. First split: Create a held-out test set (20%)
    (X_train_val, X_test, X_trials_train_val, X_trials_test, 
     y_reg_train_val, y_reg_test, y_cls_train_val, y_cls_test, 
     y_cls_label_train_val, y_cls_label_test) = train_test_split(
         *all_data, test_size=TEST_SPLIT, stratify=y_model_cls, random_state=42
    )

    y_reg_min_full = y_reg_train_val.min(axis=0)
    y_reg_max_full = y_reg_train_val.max(axis=0)

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
        
        # Initialize the model(s) once before the curriculum loop
        if config.get('is_multi_task', True):
            model = model_builder(input_shape, num_models, num_params, activation=ACTIVATION)
        else:
            cls_model, reg_model = model_builder(input_shape, num_models, num_params, activation=ACTIVATION)

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

            # Normalize regression targets using min/max of the CURRENT STAGE
            y_reg_min_stage = y_reg_train_stage.min(axis=0)
            y_reg_max_stage = y_reg_train_stage.max(axis=0)

            # Add a small epsilon to prevent division by zero for constant parameters
            y_reg_range_stage = y_reg_max_stage - y_reg_min_stage + keras.backend.epsilon()

            y_reg_train_norm_stage = (y_reg_train_stage - y_reg_min_stage) / y_reg_range_stage
            y_reg_val_norm_stage = (y_reg_val_stage - y_reg_min_stage) / y_reg_range_stage

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
            ]

            if config.get('is_multi_task', True):
                model_name = config['model_name']
                callbacks.append(ModelCheckpoint(filepath=os.path.join(MODEL_RESULTS_DIR, f"{model_name}_stage_{stage_idx}_best_model.h5"), monitor='val_loss', save_best_only=True, verbose=1))
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss=config['losses'],
                    loss_weights=config.get('loss_weights', None),
                    metrics=config['metrics']
                )

                train_targets = {'classification_output': y_cls_train_stage, config['output_names'][1]: y_reg_train_norm_stage}
                val_targets = {'classification_output': y_cls_val_stage, config['output_names'][1]: y_reg_val_norm_stage}

                model.fit(
                    x=X_train_stage, y=train_targets,
                    epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=(X_val_stage, val_targets),
                    callbacks=callbacks, verbose=1
                )
                
                model.load_weights(os.path.join(MODEL_RESULTS_DIR, f"{model_name}_stage_{stage_idx}_best_model.h5"))
                
                
            else: # Independent models
                cls_model_name = f"{config['model_name']}_Classification"
                reg_model_name = f"{config['model_name']}_Regression"
                
                # Define callbacks for both models
                cls_callbacks = [
                    EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
                    ModelCheckpoint(filepath=os.path.join(MODEL_RESULTS_DIR, f"{cls_model_name}_stage_{stage_idx}_best_model.h5"), monitor='val_loss', save_best_only=True, verbose=1)
                ]
                
                reg_callbacks = [
                    EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001),
                    ModelCheckpoint(filepath=os.path.join(MODEL_RESULTS_DIR, f"{reg_model_name}_stage_{stage_idx}_best_model.h5"), monitor='val_loss', save_best_only=True, verbose=1)
                ]
                # Train the classification model
                print("\n--- Training Independent Classification Model ---")
                cls_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=config['losses']['classification_output'], metrics=[config['metrics']['classification_output']])
                cls_model.fit(x=X_train_stage, y=y_cls_train_stage, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val_stage, y_cls_val_stage), callbacks=cls_callbacks, verbose=1)
                cls_model.load_weights(os.path.join(MODEL_RESULTS_DIR, f"{cls_model_name}_stage_{stage_idx}_best_model.h5"))

                # Train the regression model
                print("\n--- Training Independent Regression Model ---")
                reg_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=config['losses']['regression_output'], metrics=[config['metrics']['regression_output']])
                reg_model.fit(x=X_train_stage, y=y_reg_train_norm_stage, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val_stage, y_reg_val_norm_stage), callbacks=reg_callbacks, verbose=1)
                reg_model.load_weights(os.path.join(MODEL_RESULTS_DIR, f"{reg_model_name}_stage_{stage_idx}_best_model.h5"))
                
        # --- Final Evaluation on the held-out test set ---
        print("\n--- Final Evaluation on Full Test Set ---")
        
        num_conf_matrix_cols = X_proportions.shape[1]
        y_trials_test = X_trials_test
        y_reg_test_norm = (y_reg_test - y_reg_min_full) / (y_reg_max_full - y_reg_min_full + keras.backend.epsilon())

        if config.get('is_multi_task', True):
            y_pred = model.predict(X_test)
            y_pred_cls = y_pred[0]
            y_pred_reg_norm = y_pred[1]
            y_pred_reg = denormalize_predictions(y_pred_reg_norm, y_reg_min_full, y_reg_max_full)
            
            final_accuracy = accuracy_score(np.argmax(y_cls_test, axis=1), np.argmax(y_pred_cls, axis=1))
            final_mae = mean_absolute_error(y_reg_test, y_pred_reg)
            
            print(f"Final Test Accuracy: {final_accuracy:.4f}")
            print(f"Final Test MAE: {final_mae:.4f}")
            
            plot_confusion_matrix(y_cls_test, y_pred_cls, model_names, FIGURES_DIR, config['model_name'])
            plot_regression_performance(y_reg_test, y_pred_reg, y_cls_test, model_names, FIGURES_DIR, config['model_name'])
            
        else: # Independent models
            y_pred_cls = cls_model.predict(X_test)
            y_pred_reg_norm = reg_model.predict(X_test)
            y_pred_reg = denormalize_predictions(y_pred_reg_norm, y_reg_min_full, y_reg_max_full)
            
            final_accuracy = accuracy_score(np.argmax(y_cls_test, axis=1), np.argmax(y_pred_cls, axis=1))
            final_mae = mean_absolute_error(y_reg_test, y_pred_reg)

            print(f"Final Test Accuracy: {final_accuracy:.4f}")
            print(f"Final Test MAE: {final_mae:.4f}")
            
            plot_confusion_matrix(y_cls_test, y_pred_cls, model_names, FIGURES_DIR, f"{config['model_name']}_Classification")
            plot_regression_performance(y_reg_test, y_pred_reg, y_cls_test, model_names, FIGURES_DIR, f"{config['model_name']}_Regression")
