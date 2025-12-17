import tensorflow as tf
from config.model_config import penalty_lambda, batch_size

def compute_gradient_penalty(critic, real_samples, fake_samples):
    # Ensure real_samples and fake_samples have the same shape
    real_samples = tf.convert_to_tensor(real_samples)
    fake_samples = tf.convert_to_tensor(fake_samples)
    
    # Verify shapes are compatible for broadcasting
    if real_samples.shape != fake_samples.shape:
        raise ValueError(f"Incompatible shapes for broadcasting: {real_samples.shape} vs {fake_samples.shape}")
    
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    
    # Ensure alpha has the correct shape for broadcasting
    alpha = tf.broadcast_to(alpha, tf.shape(real_samples))
    
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        critic_output = critic(interpolated)
    gradients = tape.gradient(critic_output, [interpolated])[0]
    
    # Compute the gradient penalty
    gradients = tf.reshape(gradients, [batch_size, -1])
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)
    
    return gradient_penalty