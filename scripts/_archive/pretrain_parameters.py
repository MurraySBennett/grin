import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import numpy as np
import os
import importlib.util
import json
import sys
import time

# These imports are assumed to be available from your project structure
from src.utils.config import *
from src.models.independent_separate_param_loss import convert_targets_to_scales_corr

def load_model_from_file(module_name):
    """Dynamically loads the model builder and config from a specified Python file."""
    try:
        model_module = importlib.import_module(f"src.models.{module_name}")
        return getattr(model_module, f"get_{module_name.lower()}_config")()
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model from '{module_name}': {e}")
        return None, None

def train_and_evaluate_model(model, X_train, train_targets, X_val, val_targets, config, model_name):
    """Trains and evaluates a single Keras model, capturing timing and model size."""
    # Define callbacks for pre-training.
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PRE_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PRE_PATIENCE // 2, min_lr=0.00001),
        ModelCheckpoint(filepath=os.path.join(PRETRAINED_MODELS_DIR, f"{model_name}_pretrained.h5"),
                        monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # --- Start Timer ---
    start_time = time.time()
    history = model.fit(
        X_train,
        train_targets,
        epochs=PRE_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, val_targets),
        callbacks=callbacks
    )
    # --- Stop Timer ---
    end_time = time.time()
    training_time_sec = end_time - start_time
    
    # Save computational metrics
    metrics_path = os.path.join(PRETRAINED_MODEL_RESULTS_DIR, f"{model_name}_pretraining_computational_metrics.json")
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    computational_metrics = {
        'training_time_seconds': training_time_sec,
        'trainable_parameters': int(trainable_params),
        'best_epoch': int(np.argmin(history.history['val_loss'])) + 1
    }
    with open(metrics_path, 'w') as f:
        json.dump(computational_metrics, f)
    print(f"Computational metrics saved to: {metrics_path}")

    return history

if __name__ == '__main__':
    # Define the pre-training datasets to be loaded in a specific order
    PRETRAIN_DATASETS = [
        os.path.join(SIMULATED_DATA_DIR, "vary_m_dataset.npz"),
        os.path.join(SIMULATED_DATA_DIR, "vary_mv_dataset.npz"),
        os.path.join(SIMULATED_DATA_DIR, "vary_mc_dataset.npz"),
        os.path.join(SIMULATED_DATA_DIR, "vary_mvc_dataset.npz")
    ]
    
    # --- Step 1: Pre-train each model architecture with a curriculum ---
    for MODEL_FILE in MODEL_FILES:
        # Get the model builder and config from your source file
        model_builder, config = load_model_from_file(MODEL_FILE)
        if not model_builder:
            continue
            
        print(f"\n--- Starting Curriculum Pre-training for {config['model_name']} ---")
        
        # Initialize the model once before starting the curriculum
        # The input data has 16 confusion matrix values and 4 trial counts.
        input_shape = 16 + 4 
        num_models = 12
        num_params = 26
        
        if config.get('is_multi_task', True):
            model = model_builder(input_shape, num_models, num_params, dropout_rate=DROPOUT, activation=ACTIVATION)
            # Compile once with loss weights that only train the regression head
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=config['losses'],
                loss_weights={'classification_output': 0.0, config['output_names'][1]: 1.0},
                metrics=config['metrics']
            )
        else:  # Independent models
            _, reg_model = model_builder(input_shape, num_models, num_params, dropout_rate=DROPOUT, activation=ACTIVATION)

            if config['model_name'] == "independent_separate_param_loss":
                # New multi-head regression model (means, scales_corr, crit)
                reg_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss=config['losses'],
                    loss_weights=config['loss_weights'],
                    metrics=config['metrics']
                )
            else:
                # Legacy single-output regression models
                reg_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                    loss=config['losses']['regression_output'],
                    metrics=[config['metrics']['regression_output']]
                )

            model = reg_model
            
        # Initialize empty lists to hold the progressively larger dataset
        X_combined = []
        X_trials_combined = []
        y_params_combined = []
        
        # Initialize a cumulative history dictionary to store all stages
        cumulative_history = {
            key: [] for key in ['loss', 'val_loss', 'mae', 'val_mae', 'lr', 'regression_output_loss', 'val_regression_output_loss', 'regression_output_mae', 'val_regression_output_mae']
        }
        
        # Loop through each stage of the curriculum
        for stage_idx, file_path in enumerate(PRETRAIN_DATASETS):
            print(f"\n--- Stage {stage_idx + 1}: Adding data from {os.path.basename(file_path)} ---")
            if not os.path.exists(file_path):
                print(f"Error: Dataset not found at {file_path}. Please run the data generation script first.")
                sys.exit(1)
            
            # Load the data for the current stage and add it to the combined lists
            data = np.load(file_path)
            X_combined.append(data['X'])
            X_trials_combined.append(data['X_trials'])
            y_params_combined.append(data['y_params'])

            # Concatenate the data from all previous stages
            X_current = np.vstack(X_combined)
            X_trials_current = np.vstack(X_trials_combined)
            y_params_current = np.vstack(y_params_combined)

            if config['model_name'] == "independent_separate_param_loss":
                y_params_current = convert_targets_to_scales_corr(y_params_current)

            X_proportions = X_current / (np.repeat(X_trials_current, 4, axis=1) + EPSILON)
            X_trials_log = np.log(X_trials_current + EPSILON)
            X_pretrain_current = np.hstack([X_proportions, X_trials_log])
            dummy_y_cls = np.zeros((X_pretrain_current.shape[0], num_models))
            
            (X_train, X_val, y_reg_train, y_reg_val, 
             y_cls_train, y_cls_val) = train_test_split(
                X_pretrain_current, y_params_current, dummy_y_cls, test_size=0.2, random_state=42
            )

            print(f"Current stage training dataset size: {X_train.shape[0]} samples")
            print(f"Current stage validation dataset size: {X_val.shape[0]} samples")

            if config.get('is_multi_task', True):
                train_targets = {'classification_output': y_cls_train, config['output_names'][1]: y_reg_train}
                val_targets = {'classification_output': y_cls_val, config['output_names'][1]: y_reg_val}
            else:
                if config['model_name'] == "independent_separate_param_loss":
                    # Split 26-dim y_reg into (8, 12, 2)
                    y_means_train, y_scales_train, y_crit_train = (
                        y_reg_train[:, :8],
                        y_reg_train[:, 8:20],   # now 12-dim
                        y_reg_train[:, 20:]
                    )
                    y_means_val, y_scales_val, y_crit_val = (
                        y_reg_val[:, :8],
                        y_reg_val[:, 8:20],
                        y_reg_val[:, 20:]
                    )

                    train_targets = {
                        'means_out': y_means_train,
                        'scales_corr_out': y_scales_train,
                        'crit_out': y_crit_train
                    }
                    val_targets = {
                        'means_out': y_means_val,
                        'scales_corr_out': y_scales_val,
                        'crit_out': y_crit_val
                    }
                else:
                    # Old independent models
                    train_targets = y_reg_train
                    val_targets = y_reg_val

            history = train_and_evaluate_model(model, X_train, train_targets, X_val, val_targets, config, config['model_name'])
            for key, values in history.history.items():
                if key in cumulative_history:
                    cumulative_history[key].extend([float(v) for v in values])

        history_path = os.path.join(PRETRAINED_MODEL_RESULTS_DIR, f"{config['model_name']}_pretraining_history.json")
        with open(history_path, 'w') as f:
            json.dump(cumulative_history, f, indent=4)
        print(f"Full training history saved to: {history_path}")

    print("\nAll models have been successfully pre-trained and their weights saved.")
