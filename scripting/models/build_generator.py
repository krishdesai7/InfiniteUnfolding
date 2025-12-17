import tensorflow as tf
from config.model_config import gen_model_width, gen_model_depth, dropout_rate, kernel_init

def log_softplus(x):
    return tf.math.log1p(tf.nn.softplus(x))
tf.keras.utils.get_custom_objects()['log_softplus'] = log_softplus

def build_generator(input_shape):
    gen_input = tf.keras.layers.Input(shape=(input_shape,))
    x = gen_input
    
    for _ in range(gen_model_depth):
        x = tf.keras.layers.Dense(gen_model_width, use_bias=False, kernel_initializer=kernel_init if _ == 0 else None)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(1, use_bias=False, activation=log_softplus)(x)

    return tf.keras.models.Model(inputs=gen_input, outputs=outputs)