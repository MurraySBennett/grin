import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import importlib
import os
import json
import time

from src.utils.config import *

def load_model_from_file(module_name):
    """Dynamically loads the model builder and config from a specified Python file."""
    try:
        model_module = importlib.import_module(f"src.models.lstm.{module_name}")
        model_builder, config = model_module.get_model_config()
        return model_builder, config
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model from 'src.models.lstm.{module_name}': {e}")
        return None, None

def train_and_evaluate_model(model, X_train, train_targets, X_val, val_targets, config):
    """Trains and evaluates a single Keras model, capturing timing and model size."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=RLRP_FACTOR, patience=RLRP_PATIENCE, min_lr=RLRP_MIN_LR),
        ModelCheckpoint(filepath=os.path.join(MODEL_RESULTS_DIR, "lstm", f"{config['model_name']}_best_model.h5"),
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
    history_path = os.path.join(MODEL_RESULTS_DIR, "lstm", f"{config['model_name']}_history.json")
    serializable_history = {key: [float(val) for val in values] for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f)
    print(f"Training history saved to: {history_path}") 
    
    # Save computational metrics
    metrics_path = os.path.join(MODEL_RESULTS_DIR, "lstm", f"{config['model_name']}_computational_metrics.json")
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


def preprocess_data(all_trials, all_params, all_model_labels):
    """
    Prepares the data for training, ensuring all sequences are correctly shaped.
    This function explicitly checks for and handles inconsistent sequence lengths.
    """
    # Verify sequence lengths
    lengths = [len(t) for t in all_trials]
    if len(set(lengths)) > 1:
        print("Warning: Trial sequences have different lengths. Padding is required.")
        max_len = max(lengths)
    else:
        max_len = lengths[0]
        print(f"All sequences have a homogenous length of {max_len}.")

    # Pad sequences to ensure a uniform shape
    # The new error suggests that even if lengths are the same, the inner arrays
    # might be stored as Python objects. pad_sequences ensures a proper NumPy array.
    X_padded = pad_sequences(all_trials, maxlen=max_len, dtype='float32', padding='post', value=0.0)

    # Convert targets to NumPy arrays
    y_params = np.array(all_params)

    label_encoder = LabelEncoder()
    y_model_cls = label_encoder.fit_transform(all_model_labels)
    y_model_cls_categorical = to_categorical(y_model_cls)

    # Verify the final data structure
    print(f"Final preprocessed data shape: {X_padded.shape}")
    print(f"Final preprocessed data dtype: {X_padded.dtype}")

    return X_padded, y_params, y_model_cls, y_model_cls_categorical, label_encoder



if __name__ == '__main__':
    print(f"Loading pre-existing dataset from {TRIAL_BY_TRIAL_FIAL}...")
    data = np.load(TRIAL_BY_TRIAL_FIAL, allow_pickle=True)
    
    all_trials = list(data['X'])
    all_params = list(data['y_params'])
    all_model_labels = list(data['y_model_labels'])

    X, y_params, y_model_cls, y_model_cls_categorical, label_encoder = preprocess_data(all_trials, all_params, all_model_labels)

    y_model_cls = label_encoder.fit_transform(all_model_labels)
    y_model_cls_categorical = to_categorical(y_model_cls)

    (X_train_val, X_test, y_reg_train_val, y_reg_test, 
     y_cls_train_val, y_cls_test) = train_test_split(
        X, y_params, y_model_cls,
        test_size=TEST_SPLIT, stratify=y_model_cls, random_state=42
    )

    y_cls_train_val = to_categorical(y_cls_train_val)
    y_cls_test = to_categorical(y_cls_test)
    model_name_to_idx = {label: i for i, label in enumerate(label_encoder.classes_)}

    input_shape = (X.shape[1], X.shape[2])
    num_models = len(model_name_to_idx)
    num_params = y_params.shape[1]
    
    for MODEL_FILE in LSTM_MODEL_FILES:
        model_builder, config = load_model_from_file(MODEL_FILE)
        if not model_builder:
            continue

        print(f"\n--- Training {config['model_name']} ---")
        model_to_train = model_builder(input_shape, num_models, num_params, dropout_rate=DROPOUT, activation=ACTIVATION)
        model_to_train.compile(
            optimizer='adam',
            loss=config['losses'],
            metrics=config['metrics']
        )
       
        training_targets = {
            'regression_output': y_reg_train_val,
            'classification_output': y_cls_train_val
        }
        validation_targets = {
            'regression_output': y_reg_test,
            'classification_output': y_cls_test
        }
        trained_model, training_history = train_and_evaluate_model(
            model_to_train, 
            X_train_val, 
            training_targets, 
            X_test, 
            validation_targets, 
            config
        )
