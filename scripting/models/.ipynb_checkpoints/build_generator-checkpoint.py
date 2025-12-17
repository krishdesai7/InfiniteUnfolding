import tensorflow as tf
from config.model_config import gen_model_width, gen_model_depth, dropout_rate, kernel_init

def build_generator(input_shape):
    gen_input = tf.keras.layers.Input(shape=(input_shape,))
    x = gen_input
    
    for _ in range(gen_model_depth):
        x = tf.keras.layers.Dense(gen_model_width, use_bias=False, kernel_initializer=kernel_init if _ == 0 else None)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(1, use_bias=False, activation='relu')(x)

    return tf.keras.models.Model(inputs=gen_input, outputs=outputs)