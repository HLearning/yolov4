# ========================================
# @Author          : HLearning
# @Email           : hpuacm@qq.com
# @Date            : 2020-05-01
# ========================================

from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow as tf


class Mish(Layer):
    """Mish activation function.

    # Arguments
        x: Input tensor.

    # Returns
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
        Tensor, output of mish(x).

    # Examples
        X = Mish()(Input(input_shape))
    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * tf.keras.activations.tanh(tf.keras.activations.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
