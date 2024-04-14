#!/usr/bin/env python
# coding: utf-8


# +


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


# +


import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=10)
plt.rcParams["font.family"] = "serif"

from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
import pandas as pd


# +


#load and normalize the data
data = np.load('rawdata.npz')
substructure_variables = ['w', 'q', 'm', 'r', 'tau1s', 'tau2s']
data_streams = ['_true', '_true_alt', '_reco', '_reco_alt']
n_variables = len(substructure_variables)


# +


normalize = True
    
for var_name in data.files:
    globals()[var_name] = data[var_name][:150000]
    
if normalize:
    for var_name in substructure_variables:
        mu = np.mean(globals()[var_name+data_streams[0]])
        sig = np.std(globals()[var_name + data_streams[0]])
        for stream in data_streams:
            globals()[var_name+stream] = (globals()[var_name+stream] - mu)/sig


# +


N = len(m_true)

xvals_truth = np.array([np.concatenate([globals()[f"{var_name}_true_alt"], globals()[f"{var_name}_true"]]) for var_name in substructure_variables]).T
xvals_reco = np.array([np.concatenate([globals()[f"{var_name}_reco_alt"], globals()[f"{var_name}_reco"]]) for var_name in substructure_variables]).T
                    
yvals = np.concatenate([np.zeros(N, dtype=np.float32),np.ones(N, dtype=np.float32)])

X_train_truth, X_test_truth, X_train_reco, X_test_reco, Y_train, Y_test = train_test_split(
    xvals_truth, xvals_reco, yvals)


# +


tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable(package="Custom", name="weighted_wgan_critic_loss")
def weighted_wgan_critic_loss(target, output, weights):
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    output = tf.convert_to_tensor(output, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    
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


# +


# Model configuration
gen_model_width = 150
gen_model_depth = 8
disc_model_width = 75
disc_model_depth = 5
dropout_rate = 0.2
kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)


# Generator model
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

def build_critic(input_shape, disc_model_width=75, disc_model_depth=5, dropout_rate=0.2):
    critic_inputs = tf.keras.layers.Input(shape=(input_shape,))
    x = critic_inputs
    
    for _ in range(disc_model_depth):
        x = tf.keras.layers.Dense(disc_model_width, use_bias=False, kernel_initializer=kernel_init if _ == 0 else None)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        # Optionally apply batch normalization depending on its impact on stability
        # x = BatchNormalization()(x)
    
    outputs = tf.keras.layers.Dense(1, activation=None)(x)  # Linear activation for the critic output
    
    return tf.keras.models.Model(inputs=critic_inputs, outputs=outputs)

# Create the models
model_generator = build_generator(xvals_truth.shape[1])
model_critic = build_critic(xvals_reco.shape[1])



# +


def compute_gradient_penalty(critic, real_samples, fake_samples, penalty_lambda=5.0):
    batch_size = tf.shape(real_samples)[0]
    # Shape of alpha adjusted for tabular data: [batch_size, 1]
    # Ensures correct broadcasting over the feature dimension without adding unnecessary dimensions
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    
    # Interpolated samples calculation should now match the critic's expected input shape
    interpolated = real_samples + (alpha * (fake_samples - real_samples))
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = critic(interpolated, training=True)
    
    # Calculate gradients of predictions with respect to interpolated samples
    gradients = tape.gradient(predictions, [interpolated])[0]
    # Flatten the gradients to compute the norm across all dimensions except the first (batch dimension)
    gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]) + 1e-12)
    gradient_penalty = penalty_lambda * tf.reduce_mean(tf.square(gradients_norm - 1.0))
    
    return gradient_penalty


# +


def sample_real_fake():
    global X_test_detector, Y_test, batch_size
    
    real_indices = np.where(Y_test == 1)[0]
    fake_indices = np.where(Y_test == 0)[0]
    
    sampled_real_indices = np.random.choice(real_indices, size=batch_size // 2, replace=False)
    sampled_fake_indices = np.random.choice(fake_indices, size=batch_size // 2, replace=False)
    
    real_samples = X_test_reco[sampled_real_indices]
    fake_samples = X_test_reco[sampled_fake_indices]
    
    return real_samples, fake_samples


# +


initial_learning_rate = 0.0005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

optimizer_gen = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
optimizer_critic = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

@tf.function
def train_step_critic(X_detector_batch, Y_batch, W_batch):
    model_critic.trainable = True
    model_generator.trainable = False
    real_samples, fake_samples = sample_real_fake()
    
    with tf.GradientTape() as tape:
        predictions = model_critic(X_detector_batch, training=True)
        wgan_loss = weighted_wgan_critic_loss(Y_batch, predictions, W_batch)
        gradient_penalty = compute_gradient_penalty(model_critic, real_samples, fake_samples)
        total_loss = wgan_loss + gradient_penalty

    grads = tape.gradient(total_loss, model_critic.trainable_variables)
    optimizer_critic.apply_gradients(zip(grads, model_critic.trainable_variables))
    return total_loss


@tf.function(reduce_retracing=True)
def train_step_gan(X_particle_batch, X_detector_batch, Y_batch):
    model_critic.trainable = False
    model_generator.trainable = True
    with tf.GradientTape() as tape:
        W_batch = model_generator(X_particle_batch, training=True) 
        W_batch = tf.where(Y_batch == 1, 1.0, tf.squeeze(W_batch))
        critic_output = model_critic(X_detector_batch, training=False)  
        loss = -1*weighted_wgan_critic_loss(Y_batch, critic_output, W_batch)
    grads = tape.gradient(loss, model_generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(grads, model_generator.trainable_variables))
    return loss
crit_loss_avg = []
gen_loss_avg = []


# +


checkpoint_dir = './InfiniteUnfolding/model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Paths for the checkpoint files
checkpoint_path_gen = os.path.join(checkpoint_dir, 'generator_epoch-{epoch:04d}.weights.h5')

n_epochs = 1000
batch_size = X_train_reco.shape[0]//100
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reco, X_train_truth, Y_train)).batch(batch_size)

m = 1  # Number of generator updates
n = 5 # Number of critic updates

performance_metric = np.empty((n_epochs, n_variables))
data_set_1 = X_test_truth[Y_test == 1]
data_set_2 = X_test_truth[Y_test == 0]
baseline = np.array([wasserstein_distance(data_set_1[:, i], data_set_2[:, i]) for i in range(n_variables)])

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    critic_losses = []
    generator_losses = []

    for X_detector_batch, X_particle_batch, Y_batch in train_dataset:
        # Critic update loop
        for _ in range(n):
            model_generator.trainable = False
            model_critic.trainable = True
            W_batch = model_generator.predict(X_particle_batch, verbose=0)
            W_batch = tf.where(Y_batch == 1, 1.0, tf.squeeze(W_batch))
            d_loss = train_step_critic(X_detector_batch, Y_batch, W_batch)
            critic_losses.append(d_loss.numpy())

        # Generator update loop
        for _ in range(m):
            alt_indices = tf.where(Y_batch == 0)
            X_particle_alt = tf.gather_nd(X_particle_batch, alt_indices)
            X_detector_alt = tf.gather_nd(X_detector_batch, alt_indices)
            Y_alt = tf.gather_nd(Y_batch, alt_indices)
            g_loss = train_step_gan(X_particle_alt, X_detector_alt, Y_alt)
            generator_losses.append(g_loss.numpy())

    # Log and printing specs about the model
    avg_c_loss = np.mean(critic_losses[-n*len(train_dataset):])
    avg_g_loss = np.mean(generator_losses[-m*len(train_dataset):])
    gen_loss_avg.append(avg_g_loss)
    crit_loss_avg.append(avg_c_loss)   
    print(f"Epoch {epoch+1} completed. Critic Loss: {avg_c_loss}, Generator Loss: {avg_g_loss}")
    if (epoch + 1) % 10 == 0:
        gen_checkpoint_path = checkpoint_path_gen.format(epoch=epoch + 1)
        model_generator.save_weights(gen_checkpoint_path)
        print(f'Saved generator at epoch {epoch + 1}')
    weights = model_generator.predict(X_test_truth[Y_test == 0], verbose=0).flatten()
    performance_metric[epoch, :] = [wasserstein_distance(data_set_1[:, i], data_set_2[:, i], 
                                                          u_weights=None, v_weights=weights) 
                                    for i in range(n_variables)]

np.savez_compressed('WGAN_training_metrics_compressed.npz', crit_loss_avg=crit_loss_avg, gen_loss_avg=gen_loss_avg, baseline=baseline, performance_metric=performance_metric, X_test_truth = X_test_truth, Y_test = Y_test)



