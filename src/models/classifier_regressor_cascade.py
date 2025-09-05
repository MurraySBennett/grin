import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.utils.custom_losses import classification_loss, custom_means_loss, custom_cov_loss, custom_mae, dense_residual_block


def build_classifier_regressor_cascade_model(input_shape, num_models, dropout_rate=0.3, activation='tanh'):
    """
    Builds a cascaded model where the output of the classification task is
    fed into the regression task.
    """
    model_input = keras.Input(shape=(input_shape,), name='model_input')

    # --- Classification Path ---
    classification_backbone = layers.Dense(512, activation=activation, name='class_dense1')(model_input)
    classification_backbone = dense_residual_block(classification_backbone, 512, activation, dropout_rate)
    classification_backbone = layers.Dense(256, activation=activation, name='class_dense2')(classification_backbone)
    classification_backbone = layers.BatchNormalization()(classification_backbone)
    classification_backbone = layers.Dropout(dropout_rate)(classification_backbone)
    classification_backbone = layers.Dense(128, activation=activation, name='class_dense3')(classification_backbone)
    
    # Classification Head (outputs probabilities, which can be seen as confidence)
    classification_output = layers.Dense(
        num_models, activation='softmax', name='classification_output'
    )(classification_backbone)

    # --- Regression Path ---
    # The regression path takes the original input concatenated with the classification output.
    combined_input = layers.Concatenate(name='combined_input')([model_input, classification_output])
    
    regression_backbone = layers.Dense(512, activation=activation, name='reg_dense1')(combined_input)
    regression_backbone = dense_residual_block(regression_backbone, 512, activation, dropout_rate)
    regression_backbone = layers.Dense(256, activation=activation, name='reg_dense2')(regression_backbone)
    regression_backbone = layers.BatchNormalization()(regression_backbone)
    regression_backbone = layers.Dropout(dropout_rate)(regression_backbone)
    regression_backbone_output = layers.Dense(128, activation=activation, name='reg_dense3')(regression_backbone)

    # --- Regression Heads ---
    means_output = layers.Dense(6, activation='linear', name='means_output')(regression_backbone_output)
    cov_output = layers.Dense(4, activation='tanh', name='cov_output')(regression_backbone_output)
    crit_output = layers.Dense(2, activation='linear', name='crit_output')(regression_backbone_output)
    # regression_output = layers.Concatenate(name='regression_output')(
    #     [means_output, cov_params_output, crit_output]
    # )

    model = keras.Model(
        inputs=model_input,
        outputs=[classification_output, means_output, cov_output, crit_output] #regression_output]
    )
    return model

# This dictionary holds the configuration specific to this model
CLASSIFIER_REGRESSOR_CASCADE_CONFIG = {
    'losses': {
        'classification_output': classification_loss, #'categorical_crossentropy',
        'means_output': custom_means_loss,
        'cov_output': custom_cov_loss,
        'crit_output': 'mae'
    },
    'loss_weights': {
        'classification_output': 2.0,
        'means_output': 1.0,
        'cov_output': 1.5,
        'crit_output': 1.0
    },
    'metrics': {
        'classification_output': 'accuracy',
        'means_output': custom_mae,
        'cov_output': custom_mae,
        'crit_output': 'mae'
    },
    'is_multi_task': True,
    'model_name': 'CascadedClassifierRegressor',
    'output_names': ['classification_output', 'regression_output']
}

def get_classifier_regressor_cascade_config():
    """Returns the model builder and config."""
    return build_classifier_regressor_cascade_model, CLASSIFIER_REGRESSOR_CASCADE_CONFIG
