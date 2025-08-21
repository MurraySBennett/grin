import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate, Layer, Lambda, Dropout
from tensorflow.keras.models import Model
import numpy as np
from src.utils.get_covariance_matrices import get_covariance_matrices

# A Keras Layer that implements a single "expert" block
class Expert(Layer):
    """
    A simple expert sub-network with a few dense layers.
    """
    def __init__(self, units=64, dropout_rate=0.3, activation='tanh', name=None, **kwargs):
        super(Expert, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.dense1 = Dense(units, activation=activation)
        self.dropout1 = Dropout(dropout_rate)
        self.dense2 = Dense(units // 2, activation=activation)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        return self.dropout2(x)
    
    def get_config(self):
        config = super(Expert, self).get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation
        })
        return config

# A Keras Layer that implements the "gating" mechanism for a task
class Gate(Layer):
    """
    A simple gating network that learns to weight the outputs of experts.
    """
    def __init__(self, num_experts, dropout_rate=0.3, name=None, **kwargs):
        super(Gate, self).__init__(name=name, **kwargs)
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        self.dense = Dense(num_experts, activation='softmax')
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs):
        # The gate takes the shared bottom input and produces weights for the experts
        x = self.dense(inputs)
        return self.dropout(x)
    
    def get_config(self):
        config = super(Gate, self).get_config()
        config.update({
            "num_experts": self.num_experts,
            "dropout_rate": self.dropout_rate
        })
        return config

def build_customised_gate_control_model(input_shape, num_models, num_params, num_shared_experts=3, num_cls_experts=2, num_reg_experts=3, dropout_rate=0.3, activation='tanh'):
    """
    Constructs a Customized Gate Control (CGC) multi-task model.

    This model uses both shared experts and task-specific experts for each task,
    providing more flexibility in information sharing and mitigating negative transfer.

    Args:
        input_shape (int): The shape of the input data.
        num_models (int): The number of output classes for the classification task.
        num_params (int): The number of output parameters for the regression task.
        num_shared_experts (int): The number of expert sub-networks shared by all tasks.
        num_cls_experts (int): The number of experts dedicated to the classification task.
        num_reg_experts (int): The number of experts dedicated to the regression task.
        dropout_rate (float): The dropout rate to use in the expert and dense layers.
        activation (str): The activation function to use in the expert and dense layers.

    Returns:
        A Keras Model ready for compilation.
    """
    # Define input layer
    input_layer = Input(shape=(input_shape,), name='model_input')

    # --- Shared experts ---
    # Create the list of shared experts
    shared_experts = [Expert(name=f"shared_expert_{i}", dropout_rate=dropout_rate, activation=activation) for i in range(num_shared_experts)]
    # Connect the input to all shared experts
    shared_expert_outputs = [expert(input_layer) for expert in shared_experts]

    # --- Task-specific experts ---
    # Create the task-specific experts for the classification head
    cls_experts = [Expert(name=f"cls_expert_{i}", dropout_rate=dropout_rate, activation=activation) for i in range(num_cls_experts)]
    cls_expert_outputs = [expert(input_layer) for expert in cls_experts]
    
    # Create the task-specific experts for the regression head
    reg_experts = [Expert(name=f"reg_expert_{i}", dropout_rate=dropout_rate, activation=activation) for i in range(num_reg_experts)]
    reg_expert_outputs = [expert(input_layer) for expert in reg_experts]
    
    # --- Gates and Task Towers ---
    
    # Classification Gate
    # The classification gate sees its own experts and the shared experts
    cls_gate = Gate(num_shared_experts + num_cls_experts, dropout_rate=dropout_rate, name='cls_gate')
    cls_gate_output = cls_gate(input_layer)
    
    # Stack the expert outputs to prepare for a weighted sum
    cls_stacked_experts = tf.stack(cls_expert_outputs + shared_expert_outputs, axis=1)
    # The gate weights the concatenated expert outputs
    cls_tower_input = Lambda(lambda x: tf.einsum('bi,bij->bj', x[1], x[0]))(
        [cls_stacked_experts, cls_gate_output]
    )
    
    # Classification Tower
    cls_tower = Dense(128, activation=activation)(cls_tower_input)
    cls_tower = Dropout(dropout_rate)(cls_tower)
    classification_output = Dense(num_models, activation='softmax', name='classification_output')(cls_tower)

    # Regression Gate
    # The regression gate sees its own experts and the shared experts
    reg_gate = Gate(num_shared_experts + num_reg_experts, dropout_rate=dropout_rate, name='reg_gate')
    reg_gate_output = reg_gate(input_layer)

    # Stack the expert outputs to prepare for a weighted sum
    reg_stacked_experts = tf.stack(reg_expert_outputs + shared_expert_outputs, axis=1)
    # The gate weights the concatenated expert outputs
    reg_tower_input = Lambda(lambda x: tf.einsum('bi,bij->bj', x[1], x[0]))(
        [reg_stacked_experts, reg_gate_output]
    )
    
    # --- Regression Head with separate outputs ---
    reg_head = Dense(128, activation=activation)(reg_tower_input)
    reg_head = Dropout(dropout_rate)(reg_head)

    # We need to predict the Cholesky parameters separately (12 parameters)
    num_chol_params = 12
    chol_params_output = Dense(num_chol_params, activation='linear', name='chol_params_output')(reg_head)
    
    # Use the custom function to convert Cholesky parameters to covariance matrices
    cov_matrices_output = Lambda(
        lambda chol_params: get_covariance_matrices(chol_params),
        name='cov_matrices_output'
    )(chol_params_output)
    
    # Predict the means (8 parameters)
    means_output = Dense(8, activation='linear', name='means_output')(reg_head)

    # Predict the critical values (2 parameters)
    crit_output = Dense(2, activation='linear', name='crit_output')(reg_head)

    # Combine all regression outputs into a single tensor
    regression_output = Concatenate(name='regression_output')(
        [means_output, cov_matrices_output, crit_output]
    )
    
    # Final model
    model = Model(
        inputs=input_layer,
        outputs=[classification_output, regression_output]
    )

    return model

CUSTOMISED_GATE_CONTROL_CONFIG = {
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
    'is_multi_task': True,
    'model_name': 'CustomisedGateControl',
    'output_names': ['classification_output', 'regression_output']
}

def get_customised_gate_control_model_config():
    """Returns the model builder and config."""
    return build_customised_gate_control_model, CUSTOMISED_GATE_CONTROL_CONFIG
