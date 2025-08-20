import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.get_covariance_matrices import get_covariance_matrices

def build_cascaded_non_bayesian_model(input_shape, num_models, num_params, dropout_rate=0.3, activation='tanh'):
    """
    Builds a cascaded multi-task learning model. A shared backbone feeds into a
    standard regression head, and its predictions are then used as features for the
    classification head.
    
    Args:
        input_shape (int): The shape of the input data (flattened confusion matrix).
        num_models (int): The number of GRT models to classify.
        num_params (int): The number of parameters to regress.
        dropout_rate (float): Dropout rate for regularization.
        
    Returns:
        keras.Model: The compiled cascaded multi-task model.
    """
    model_input = keras.Input(shape=(input_shape,), name='model_input')

    # --- Shared Feature Backbone ---
    shared_backbone = layers.Dense(512, activation=activation, name='shared_dense1')(model_input)
    shared_backbone = layers.BatchNormalization()(shared_backbone)
    shared_backbone = layers.Dropout(dropout_rate)(shared_backbone)
    shared_backbone = layers.Dense(256, activation=activation, name='shared_dense2')(shared_backbone)
    shared_backbone = layers.BatchNormalization()(shared_backbone)
    shared_backbone = layers.Dropout(dropout_rate)(shared_backbone)

    # --- Standard Regression Head ---
    # We need to predict parameters for all 4 covariance matrices
    num_cov_params = 12
    chol_params_output = layers.Dense(num_cov_params, activation='linear', name='chol_params_output')(shared_backbone)
    cov_matrices_output = layers.Lambda(
        lambda chol_params: get_covariance_matrices(chol_params),
        name='cov_matrices_output'
    )(chol_params_output) 
 
    # --- Output Head for Means (8 parameters) ---
    # The first 8 parameters in your target data correspond to the means
    means_output = layers.Dense(8, activation='linear', name='means_output')(shared_backbone)

    # --- Output Head for Critical Values (2 parameters) ---
    # The last 2 parameters in your target data correspond to the 'crit' values
    crit_output = layers.Dense(2, activation='linear', name='crit_output')(shared_backbone)

    # --- Concatenate all outputs ---
    # Combine all three output heads into a single tensor of shape (batch_size, 26)
    regression_output = layers.Concatenate(name='regression_output')(
        [means_output, cov_matrices_output, crit_output]
    )
    
    # Concatenate the original backbone features with the regression predictions
    classifier_input = layers.Concatenate()([shared_backbone, regression_output])

    # --- Cascaded Classification Head ---
    cls_head = layers.Dense(128, activation=activation, name='cls_dense1_cascaded')(classifier_input)
    cls_head = layers.BatchNormalization()(cls_head)
    cls_head = layers.Dropout(dropout_rate)(cls_head)
    classification_output = layers.Dense(
        num_models, activation='softmax', name='classification_output'
    )(cls_head)

    model = keras.Model(
        inputs=model_input,
        outputs=[classification_output, regression_output]
    )
    return model

# This dictionary holds the configuration specific to this model
CASCADED_NON_BAYESIAN_CONFIG = {
    'losses': {
        'classification_output': 'categorical_crossentropy',
        'regression_output': 'mean_squared_error'
    },
    'loss_weights': {
        'classification_output': 1.0,
        'regression_output': 1.0
    },
    'metrics': {
        'classification_output': 'accuracy',
        'regression_output': 'mae'
    },
    'is_bayesian': False,
    'model_name': 'CascadedNonBayesian',
    'output_names': ['classification_output', 'regression_output']
}

def get_cascaded_non_bayesian_model_config():
    """Returns the model builder and config."""
    return build_cascaded_non_bayesian_model, CASCADED_NON_BAYESIAN_CONFIG
