# gru_model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Lambda, concatenate
from tensorflow.keras.models import Model
from src.utils.param_training_funcs import get_covariance_matrices

def build_model(input_shape, num_models, num_params, dropout_rate=0.3, activation='relu'):
    """
    Builds a multi-task GRU model with a shared backbone and two output heads.
    One head is for regression, and the other for classification.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        num_models (int): The number of unique model classes for the classification task.
        num_params (int): The number of outputs for the full regression task.
        dropout_rate (float): The dropout rate for regularization.
        activation (str): The activation function for the dense layers.

    Returns:
        A Keras Model instance.
    """
    # Define the input layer with the sequence shape
    model_input = Input(shape=input_shape, name='model_input')

    # --- Shared GRU Backbone ---
    # The GRU layer processes the entire sequence and returns the final hidden state.
    gru_backbone = GRU(units=64, return_sequences=False, name='gru_core')(model_input)
    gru_backbone = Dense(units=64, activation=activation)(gru_backbone)
    gru_backbone = Dropout(dropout_rate)(gru_backbone)
    gru_backbone = Dense(units=32, activation=activation)(gru_backbone)
    gru_backbone = Dropout(dropout_rate)(gru_backbone)

    # --- Classification Head ---
    classification_output = Dense(num_models, activation='softmax', name='classification_output')(gru_backbone)

    # --- Regression Head ---
    means_outputs = Dense(8, activation='linear', name='mean_outputs')(gru_backbone)
    cov_inputs = Dense(12, activation='linear', name='chol_params_raw')(gru_backbone)
    cov_outputs = Lambda(get_covariance_matrices, name='cov_matrix_output')(cov_inputs)
    crit_outputs = Dense(2, activation='linear', name='crit_output')(gru_backbone)

    # Concatenate the outputs
    regression_output = concatenate([means_outputs, cov_outputs, crit_outputs], name='regression_output')
    
    # --- Combine and build the final model ---
    model = Model(
        inputs=model_input,
        outputs=[classification_output, regression_output]
    )

    return model

# Configuration dictionary for the model
def get_model_config():
    return build_model, {
        'losses': {'regression_output': 'mse', 'classification_output': 'categorical_crossentropy'},
        'metrics': {'regression_output': 'mae', 'classification_output': 'accuracy'},
        'model_name': 'GRU_MultiTask'
    }
