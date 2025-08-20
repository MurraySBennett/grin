import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.get_covariance_matrices import get_covariance_matrices


def build_classification_model(input_shape, num_models, dropout_rate=0.3, activation='tanh'):
    """
    Builds a standalone classification model, now including trials_input.
    """
    # confusion_matrix_input = keras.Input(shape=(input_shape,), name='cls_input')
    # trials_input = keras.Input(shape=(trials_input_shape,), name='trials_input')
    # combined_input = layers.Concatenate(name='combined_input')([confusion_matrix_input, trials_input])
    model_input = keras.Input(shape=(input_shape,), name='model_input')
    
    x = layers.Dense(512, activation=activation, name='cls_dense1')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation=activation, name='cls_dense2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    classification_output = layers.Dense(num_models, activation='softmax', name='classification_output')(x)
    
    return keras.Model(
        inputs=model_input, #[confusion_matrix_input, trials_input],
        outputs=classification_output
    )

def build_regression_model(input_shape, num_params, dropout_rate=0.3, activation='tanh'):
    """
    Builds a standalone regression model
    """
    input_layer = keras.Input(shape=(input_shape,), name='reg_input')
    x = layers.Dense(512, activation=activation, name='reg_dense1')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation=activation, name='reg_dense2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Predict the parameters for the lower triangular matrices
    # For 4 covariance matrices, each with 3 unique parameters: 4 * 3 = 12 parameters
    num_cov_params = 12
    chol_params_output = layers.Dense(num_cov_params, activation='linear', name='chol_params_output')(x)
    cov_matrices_output = layers.Lambda(
        lambda chol_params: get_covariance_matrices(chol_params),
        name='cov_matrices_output'
    )(chol_params_output) 

    # --- Output Head for Means (8 parameters) ---
    # The first 8 parameters in your target data correspond to the means
    means_output = layers.Dense(8, activation='linear', name='means_output')(x)

    # --- Output Head for Critical Values (2 parameters) ---
    # The last 2 parameters in your target data correspond to the 'crit' values
    crit_output = layers.Dense(2, activation='linear', name='crit_output')(x)

    # --- Concatenate all outputs ---
    # Combine all three output heads into a single tensor of shape (batch_size, 26)
    regression_output = layers.Concatenate(name='regression_output')(
        [means_output, cov_matrices_output, crit_output]
    )
    
    return keras.Model(inputs=input_layer, outputs=regression_output)




def build_independent_models(input_shape, num_models, num_params, activation='tanh'):
    """
    This function is a placeholder for your generalized training script.
    It returns the two separate models.
    """
    cls_model = build_classification_model(input_shape, num_models, activation=activation)
    reg_model = build_regression_model(input_shape, num_params, activation=activation)
    return cls_model, reg_model

INDEPENDENT_CONFIG = {
    'losses': {
        'classification_output': 'categorical_crossentropy',
        'regression_output': 'mean_squared_error'
    },
    'metrics': {
        'classification_output': 'accuracy',
        'regression_output': 'mae'
    },
    'is_multi_task': False, # Flag to indicate independent training
    'model_name': 'Independent'
}

def get_independent_models_config():
    """Returns the model builder and config."""
    return build_independent_models, INDEPENDENT_CONFIG

