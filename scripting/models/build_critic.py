import tensorflow as tf
from config.model_config import disc_model_width, disc_model_depth, dropout_rate, kernel_init

def build_critic(input_shape, disc_model_width=disc_model_width, disc_model_depth=disc_model_depth, dropout_rate=dropout_rate):
    critic_inputs = tf.keras.layers.Input(shape=(input_shape,))
    x = critic_inputs
    
    for _ in range(disc_model_depth):
        x = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Dense(disc_model_width, use_bias=False, kernel_initializer=kernel_init if _ == 0 else None)
            )(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.LayerNormalization()(x)
    
    outputs = tf.keras.layers.SpectralNormalization(
                tf.keras.layers.Dense(1, activation=None)
            )(x)
    
    return tf.keras.models.Model(inputs=critic_inputs, outputs=outputs)