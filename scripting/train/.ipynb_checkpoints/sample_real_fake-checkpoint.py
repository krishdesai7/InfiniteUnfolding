from config.model_config import batch_size
import tensorflow as tf

def sample_real_fake(X_train_reco, Y_train):
    # Get indices of real and fake samples
    real_indices = tf.where(Y_train == 1)
    fake_indices = tf.where(Y_train == 0)
    
    real_indices = tf.squeeze(real_indices)
    fake_indices = tf.squeeze(fake_indices)
    
    # Shuffle the indices
    real_indices = tf.random.shuffle(real_indices)
    fake_indices = tf.random.shuffle(fake_indices)
    
    # Determine the number of samples we can actually draw
    num_real_samples = tf.minimum(tf.shape(real_indices)[0], batch_size // 2)
    num_fake_samples = tf.minimum(tf.shape(fake_indices)[0], batch_size // 2)
    
    # Select a subset of indices based on available samples
    sampled_real_indices = real_indices[:num_real_samples]
    sampled_fake_indices = fake_indices[:num_fake_samples]
    
    # Gather the corresponding samples
    real_samples = tf.gather(X_train_reco, sampled_real_indices)
    fake_samples = tf.gather(X_train_reco, sampled_fake_indices)
    
    # Pad samples if necessary to ensure consistent batch size
    real_samples = tf.pad(real_samples, [[0, batch_size // 2 - tf.shape(real_samples)[0]], [0, 0]])
    fake_samples = tf.pad(fake_samples, [[0, batch_size // 2 - tf.shape(fake_samples)[0]], [0, 0]])

    return real_samples, fake_samples