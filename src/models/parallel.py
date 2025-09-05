import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.custom_losses import classification_loss, custom_means_loss, custom_cov_loss, custom_mae, dense_residual_block

def build_parallel_model(input_shape, num_models, dropout_rate=0.3, activation='tanh'):
    """
    Builds a multi-task learning model with two separate backbones 
    (one for classification and one for regression).
    """
    model_input = keras.Input(shape=(input_shape,), name='model_input')

    # --- Classification Feature Backbone ---
    cls_backbone = layers.Dense(512, activation=activation, name='cls_dense1')(model_input)
    cls_backbone = dense_residual_block(cls_backbone, 512, activation, dropout_rate)
    cls_backbone = layers.Dense(256, activation=activation, name='cls_dense2')(cls_backbone)
    cls_backbone = layers.BatchNormalization()(cls_backbone)
    cls_backbone = layers.Dropout(dropout_rate)(cls_backbone)
    cls_backbone = layers.Dense(128, activation=activation, name='cls_dense3')(cls_backbone)

    # --- Classification Head ---
    classification_output = layers.Dense(
        num_models, activation='softmax', name='classification_output'
    )(cls_backbone)

    # --- Regression Feature Backbone ---
    reg_backbone = layers.Dense(512, activation=activation, name='reg_dense1')(model_input)
    reg_backbone = dense_residual_block(reg_backbone, 512, activation, dropout_rate)
    reg_backbone = layers.Dense(256, activation=activation, name='reg_dense2')(reg_backbone)
    reg_backbone = layers.BatchNormalization()(reg_backbone)
    reg_backbone = layers.Dropout(dropout_rate)(reg_backbone)
    reg_backbone = layers.Dense(128, activation=activation, name='reg_dense3')(reg_backbone)

    # --- Output Heads for Regression ---
    means_output = layers.Dense(6, activation='linear', name='means_output')(reg_backbone)
    cov_output = layers.Dense(4, activation='tanh', name='cov_output')(reg_backbone)
    crit_output = layers.Dense(2, activation='linear', name='crit_output')(reg_backbone)
    # regression_output = layers.Concatenate(name='regression_output')(
    #     [means_output, cov_params_output, crit_output]
    # )

    model = keras.Model(
        inputs=model_input,
        outputs=[classification_output, means_output, cov_output, crit_output] #regression_output]
    )
    return model

# This dictionary holds the configuration specific to this model
PARALLEL_CONFIG = {
    'losses': {
        'classification_output': classification_loss, # 'categorical_crossentropy',
        'means_output': custom_means_loss,
        'cov_output': custom_cov_loss,
        'crit_output': 'mae'
    },
    'loss_weights': {
        'classification_output': 1.0,
        'means_output': 1.0,
        'cov_output': 2.0,
        'crit_output': 1.0
    },
    'metrics': {
        'classification_output': 'accuracy',
        'means_output': custom_mae,
        'cov_output': custom_mae,
        'crit_output': 'mae'
    },
    'is_multi_task': True,
    'model_name': 'ParallelMultiTask',
    'output_names': ['classification_output', 'regression_output']
}

def get_parallel_config():
    """Returns the model builder and config."""
    return build_parallel_model, PARALLEL_CONFIG
