import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="weighted_wgan_critic_loss")
def weighted_wgan_critic_loss(target, output, weights):  
    
    target = tf.cast(target, weights.dtype)

    # Separate weights for real and fake samples
    weights_real = weights * target
    weights_fake = weights * (1 - target)
    
    # Calculate the weighted scores for real and fake samples
    weighted_scores_real = output * weights_real
    weighted_scores_fake = output * weights_fake
    
    # Calculate the sum of weights for normalization
    sum_weights_real = tf.reduce_sum(weights_real) + tf.keras.backend.epsilon()
    sum_weights_fake = tf.reduce_sum(weights_fake) + tf.keras.backend.epsilon()
    
    # Calculate the weighted average scores
    weighted_average_real = tf.reduce_sum(weighted_scores_real) / sum_weights_real
    weighted_average_fake = tf.reduce_sum(weighted_scores_fake) / sum_weights_fake
    
    # WGAN critic loss is the difference between weighted averages
    wgan_critic_loss = weighted_average_fake - weighted_average_real
    
    return wgan_critic_loss