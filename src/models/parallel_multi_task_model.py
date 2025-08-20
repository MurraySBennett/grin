import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.get_covariance_matrices import get_covariance_matrices

def build_parallel_multi_task_model(input_shape, num_models, num_params, activation='tanh'):
    """Builds a multi-task learning model with separate feature backbones."""
    model_input = keras.Input(shape=(input_shape,), name='model_input')

    # --- Classification Feature Backbone ---
    cls_backbone = layers.Dense(512, activation=activation, name='cls_dense1')(model_input)
    cls_backbone = layers.BatchNormalization()(cls_backbone)
    cls_backbone = layers.Dropout(0.3)(cls_backbone)
    cls_backbone = layers.Dense(256, activation=activation, name='cls_dense2')(cls_backbone)
    cls_backbone = layers.BatchNormalization()(cls_backbone)
    cls_backbone = layers.Dropout(0.3)(cls_backbone)
    cls_backbone = layers.Dense(128, activation=activation, name='cls_dense3')(cls_backbone)

    # --- Classification Head ---
    classification_output = layers.Dense(
        num_models, activation='softmax', name='classification_output'
    )(cls_backbone)

    # --- Regression Feature Backbone ---
    reg_backbone = layers.Dense(512, activation=activation, name='reg_dense1')(model_input)
    reg_backbone = layers.BatchNormalization()(reg_backbone)
    reg_backbone = layers.Dropout(0.3)(reg_backbone)
    reg_backbone = layers.Dense(256, activation=activation, name='reg_dense2')(reg_backbone)
    reg_backbone = layers.BatchNormalization()(reg_backbone)
    reg_backbone = layers.Dropout(0.3)(reg_backbone)
    reg_backbone = layers.Dense(128, activation=activation, name='reg_dense3')(reg_backbone)

    num_chol_params = 12
    chol_params_output = layers.Dense(num_chol_params, activation='linear', name='chol_params_output')(reg_backbone)
    cov_matrices_output = layers.Lambda(
        lambda chol_params: get_covariance_matrices(chol_params),
        name='cov_matrices_output'
    )(chol_params_output) 
        
    # --- Output Head for Means (8 parameters) ---
    # The first 8 parameters in your target data correspond to the means
    means_output = layers.Dense(8, activation='linear', name='means_output')(reg_backbone)

    # --- Output Head for Critical Values (2 parameters) ---
    # The last 2 parameters in your target data correspond to the 'crit' values
    crit_output = layers.Dense(2, activation='linear', name='crit_output')(reg_backbone)

    # --- Concatenate all outputs ---
    # Combine all three output heads into a single tensor of shape (batch_size, 26)
    regression_output = layers.Concatenate(name='regression_output')(
        [means_output, cov_matrices_output, crit_output]
    )

    model = keras.Model(
        inputs=model_input,
        outputs=[classification_output, regression_output]
    )
    return model


# This dictionary holds the configuration specific to this model
PARALLEL_MULTI_TASK_CONFIG = {
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
    'model_name': 'ParallelMultiTask',
    'output_names': ['classification_output', 'regression_output']
}

def get_parallel_multi_task_model_config():
    """Returns the model builder and config."""
    return build_parallel_multi_task_model, PARALLEL_MULTI_TASK_CONFIG
