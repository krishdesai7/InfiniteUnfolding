import tensorflow as tf
from models.custom_loss import weighted_wgan_critic_loss
from train.gradient_penalty import compute_gradient_penalty
from train.sample_real_fake import sample_real_fake

def train_step_critic(model_critic, model_generator, X_detector_batch, Y_batch, W_batch, optimizer_critic):
    model_critic.trainable = True
    model_generator.trainable = False
    real_samples, fake_samples = sample_real_fake(X_detector_batch, Y_batch)
    
    with tf.GradientTape() as tape:
        predictions = model_critic(X_detector_batch, training=True)
        wgan_loss = weighted_wgan_critic_loss(Y_batch, predictions, W_batch)
        gradient_penalty = compute_gradient_penalty(model_critic, real_samples, fake_samples)
        total_loss = wgan_loss + gradient_penalty

    grads = tape.gradient(total_loss, model_critic.trainable_variables)
    optimizer_critic.apply_gradients(zip(grads, model_critic.trainable_variables))
    return total_loss

@tf.function(reduce_retracing=True)
def train_step_gan(model_critic, model_generator, X_particle_batch, X_detector_batch, Y_batch, optimizer_gen):
    model_critic.trainable = False
    model_generator.trainable = True
    with tf.GradientTape() as tape:
        W_batch = model_generator(X_particle_batch, training=True) 
        W_batch = tf.where(Y_batch == 1, 1.0, tf.squeeze(W_batch))
        critic_output = model_critic(X_detector_batch, training=False)  
        loss = -1e-3 * weighted_wgan_critic_loss(Y_batch, critic_output, W_batch)
    grads = tape.gradient(loss, model_generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(grads, model_generator.trainable_variables))
    return loss