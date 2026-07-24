import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from pprint import pprint as pp

from src.utils.config import *


def plot_confusion_matrix(y_true_labels, y_pred_labels, all_model_names, save_dir, model_name):
    """
    Plots a confusion matrix for model classifications using string labels.
    """
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=all_model_names)
    cm_df = pd.DataFrame(cm, index=all_model_names, columns=all_model_names)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized_df = pd.DataFrame(cm_normalized, index=all_model_names, columns=all_model_names)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized_df, annot=True, fmt='.2f', cmap='Blues', xticklabels=all_model_names, yticklabels=all_model_names)
    plt.title(f'Normalized Confusion Matrix for {model_name} Classification', fontsize=16)
    plt.ylabel('True Model')
    plt.xlabel('Predicted Model')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'class_confusions_{model_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")
    
def plot_overall_performance(accuracy, save_dir, model_name):
    """
    Plots a simple bar chart of the overall classification accuracy.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy'], [accuracy * 100], color='skyblue')
    plt.ylim(0, 100)
    plt.ylabel('Percentage')
    plt.title(f'Overall Classification Accuracy for {model_name}', fontsize=16)
    plt.text(0, accuracy * 100 + 1, f'{accuracy * 100:.2f}%', ha='center')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'overall_performance_{model_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Overall performance plot saved to: {save_path}")

def build_ensemble_classifier(input_shape, num_classes):
    """
    Builds a neural network classifier that takes concatenated parameters as input.
    """
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    # --- Data Loading and Preprocessing ---
    print("Loading full dataset...")
    data = np.load(DATASET_FILE, allow_pickle=True)
    y_cls_name = data['y_cls_label']
    
    print(f"Loading matrix features from {MATRIX_FEATURE_FILE}...")
    feature_data = np.load(MATRIX_FEATURE_FILE)
    X_input = feature_data['X_input']
    
    # --- Label Encoding ---
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_cls_name)
    num_classes = len(np.unique(integer_encoded))
    y_one_hot = to_categorical(integer_encoded, num_classes=num_classes)
    
    # --- Split dataset (train vs test) ---
    train_val_idx, test_idx = train_test_split(
        np.arange(len(y_cls_name)),
        test_size=TEST_SPLIT,
        stratify=y_cls_name,
        random_state=42
    )

    X_train_raw_features = X_input[train_val_idx]
    y_train_raw_labels = y_one_hot[train_val_idx]

    X_test_raw_features = X_input[test_idx]
    y_test_raw_labels = y_one_hot[test_idx]
    
    y_cls_name_test = y_cls_name[test_idx]
    
    # --- Load Specialist Models and Generate Feature Vectors ---
    print("\nLoading specialist models and generating parameter predictions...")
    specialist_models = {}
    for model_name in MODEL_NAMES:
        model_path = os.path.join(MODEL_RESULTS_DIR, f"specialist_model_{model_name}.h5")
        if os.path.exists(model_path):
            specialist_models[model_name] = tf.keras.models.load_model(model_path, compile=False)

    def get_concatenated_predictions(data_features):
        all_preds = []
        for model_name, model in specialist_models.items():
            preds = model.predict(data_features, verbose=0)
            
            # Concatenate all outputs for the current model
            model_preds = []
            for output_name in ['means_output', 'cov_output', 'crit_output']:
                if output_name in preds:
                    model_preds.append(preds[output_name])
            
            # Pad the predictions to a consistent size for models with fewer parameters
            # and reshape for concatenation
            if model_name in ['ps_ds', 'psa_ds', 'psb_ds']:
                if model_name == 'ps_ds':
                    padded_means = np.zeros((preds['means_output'].shape[0], 6))
                    padded_means[:, [0, 3]] = preds['means_output']
                    preds['means_output'] = padded_means
                elif model_name == 'psa_ds':
                    padded_means = np.zeros((preds['means_output'].shape[0], 6))
                    padded_means[:, [0, 1, 3, 5]] = preds['means_output']
                    preds['means_output'] = padded_means
                elif model_name == 'psb_ds':
                    padded_means = np.zeros((preds['means_output'].shape[0], 6))
                    padded_means[:, [0, 2, 3, 4]] = preds['means_output']
                    preds['means_output'] = padded_means
            
            # Handle single vs. multi-output for covariance
            if 'rho1' in model_name:
                preds['cov_output'] = np.hstack([preds['cov_output']]*4) # Stack 4 times to match other models
            
            # Ensure output dimensions are correct before hstacking
            means_out = preds['means_output']
            crit_out = preds['crit_output']
            
            if 'pi' in model_name:
                concatenated_preds = np.hstack([means_out, crit_out])
            else:
                cov_out = preds['cov_output']
                concatenated_preds = np.hstack([means_out, cov_out, crit_out])
            
            all_preds.append(concatenated_preds)
            
        return np.hstack(all_preds)

    X_train_ensemble = get_concatenated_predictions(X_train_raw_features)
    X_test_ensemble = get_concatenated_predictions(X_test_raw_features)
    
    # --- Build and Train the Ensemble Classifier ---
    input_shape_ensemble = X_train_ensemble.shape[1]
    ensemble_classifier = build_ensemble_classifier(input_shape_ensemble, num_classes)
    
    print("\nTraining the ensemble classifier...")
    history = ensemble_classifier.fit(
        X_train_ensemble,
        y_train_raw_labels,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )

    # --- Evaluate Classification Performance ---
    print("\nEvaluating the ensemble classifier on th class test set...")
    loss, accuracy = ensemble_classifier.evaluate(X_test_ensemble, y_test_raw_labels, verbose=0)
    print(f"\nFinal Ensemble Classifier Accuracy: {accuracy:.4f}")

    y_pred_probs = ensemble_classifier.predict(X_test_ensemble)
    y_pred_labels_encoded = np.argmax(y_pred_probs, axis=1)
    y_true_labels_encoded = np.argmax(y_test_raw_labels, axis=1)

    y_pred_labels = label_encoder.inverse_transform(y_pred_labels_encoded)
    y_true_labels = label_encoder.inverse_transform(y_true_labels_encoded)
    
    # Plotting results
    all_model_names = label_encoder.classes_
    plot_confusion_matrix(
        y_true_labels=y_true_labels,
        y_pred_labels=y_pred_labels,
        all_model_names=all_model_names,
        save_dir=os.path.join(FIGURES_DIR, "specialist_classifier"),
        model_name="Ensemble_Classifier"
    )
    
    plot_overall_performance(accuracy, os.path.join(FIGURES_DIR, "specialist_classifier"), "Ensemble_Classifier")
