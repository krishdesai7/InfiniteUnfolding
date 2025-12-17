import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=10)
plt.rcParams["font.family"] = "serif"

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization, concatenate, Layer, Lambda, Multiply, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
import pandas as pd

#load and normalize the data
data = np.load('rawdata.npz')
substructure_variables = ['w', 'q', 'm', 'r', 'tau1s', 'tau2s']
data_streams = ['_true', '_true_alt', '_reco', '_reco_alt']
n_variables = len(substructure_variables)


normalize = True
    
for var_name in data.files:
    globals()[var_name] = data[var_name][:150000]
    
if normalize:
    for var_name in substructure_variables:
        mu = np.mean(globals()[var_name+data_streams[0]])
        sig = np.std(globals()[var_name + data_streams[0]])
        for stream in data_streams:
            globals()[var_name+stream] = (globals()[var_name+stream] - mu)/sig

N = len(m_true)

xvals_truth = np.array([np.concatenate([globals()[f"{var_name}_true_alt"], globals()[f"{var_name}_true"]]) for var_name in substructure_variables]).T
xvals_reco = np.array([np.concatenate([globals()[f"{var_name}_reco_alt"], globals()[f"{var_name}_reco"]]) for var_name in substructure_variables]).T
                    
yvals = np.concatenate([np.zeros(N, dtype=np.float32),np.ones(N, dtype=np.float32)])

X_train_truth, X_test_truth, X_train_reco, X_test_reco, Y_train, Y_test = train_test_split(
    xvals_truth, xvals_reco, yvals)

tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable(package="Custom", name="weighted_binary_crossentropy")
def weighted_binary_crossentropy(target, output, weights):
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    output = tf.convert_to_tensor(output, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    
    epsilon_ = tf.keras.backend.epsilon()
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    
    weights_1 = tf.reduce_sum(target * weights)
    weights_0 = tf.reduce_sum((1 - target) * weights)
    
    bce_1 = target * tf.math.log(output + epsilon_) * weights / weights_1
    bce_0 = (1 - target) * tf.math.log(1 - output + epsilon_) * weights / weights_0
    weighted_bce = -tf.reduce_mean(bce_1 + bce_0) * tf.cast(tf.shape(target)[0], dtype=tf.float32)
    return weighted_bce
    
@tf.keras.utils.register_keras_serializable(package="Custom", name="weighted_binary_crossentropy_GAN")
def weighted_binary_crossentropy_GAN(target, output, weights):
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    output = tf.convert_to_tensor(output, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)

    epsilon_ = tf.keras.backend.epsilon()
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    
    weights_sum = tf.reduce_sum((1 - target) * weights)
    bce = weights * (1 - target) * tf.math.log(1 - output + epsilon_) / weights_sum
    weighted_bce = 2 * tf.reduce_mean(bce) * tf.cast(tf.shape(target)[0], dtype=tf.float32)
    return weighted_bce
# Model configuration
gen_model_width = 100
gen_model_depth = 5
disc_model_width = 50
disc_model_depth = 3
dropout_rate = 0.0

# Generator model
def build_generator(input_shape):
    gen_input = Input(shape=(input_shape,))
    x = gen_input
    
    # Initialize kernel
    kernel_init = RandomNormal(mean=0.0, stddev=0.2)
    
    # Hidden layers
    for _ in range(gen_model_depth):
        x = Dense(gen_model_width, activation='relu', use_bias=False, kernel_initializer=kernel_init if _ == 0 else None)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    # Output layer
    x = Dense(1, activation='softplus', use_bias=False)(x)
    outputs = Lambda(lambda x: x / K.log(2.0))(x)
    return Model(inputs=gen_input, outputs=outputs)

# Discriminator model
def build_critic(input_shape):
    critic_inputs = Input(shape=(input_shape,))
    
    # Hidden layers
    x = critic_inputs
    for _ in range(disc_model_depth):
        x = Dense(disc_model_width, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.2) if _ == 0 else None)(x)
    
    # Output layer - Note the removal of the sigmoid activation
    outputs = Dense(1, activation=None)(x)
    
    return Model(inputs=critic_inputs, outputs=outputs)

# Create the models
model_generator = build_generator(xvals_truth.shape[1])
model_discriminator = build_discriminator(xvals_reco.shape[1])

optimizer_disc = Adam(learning_rate=0.0001, beta_1 = 0.3)
optimizer_gen = Adam(learning_rate=0.0002, beta_1 = 0.5)

@tf.function
def train_step_discriminator(X_detector_batch, Y_batch, W_batch):
    model_discriminator.trainable = True
    model_generator.trainable = False
    with tf.GradientTape() as tape:
        predictions = model_discriminator(X_detector_batch, training=True)
        loss = weighted_binary_crossentropy(Y_batch, predictions, W_batch)
    grads = tape.gradient(loss, model_discriminator.trainable_variables)
    optimizer_disc.apply_gradients(zip(grads, model_discriminator.trainable_variables))
    return loss

@tf.function
def train_step_gan(X_particle_batch, X_detector_batch, Y_batch):
    model_discriminator.trainable = False
    model_generator.trainable = True
    with tf.GradientTape() as tape:
        W_batch = model_generator(X_particle_batch, training=True) 
        W_batch = tf.where(Y_batch == 1, 1.0, tf.squeeze(W_batch))
        discriminator_output = model_discriminator(X_detector_batch, training=False)  
        loss = weighted_binary_crossentropy_GAN(Y_batch, discriminator_output, W_batch)
    grads = tape.gradient(loss, model_generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(grads, model_generator.trainable_variables))
    return loss
    
disc_loss_avg = []
gen_loss_avg = []

checkpoint_dir = './InfiniteUnfolding/model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Paths for the checkpoint files
checkpoint_path_gen = os.path.join(checkpoint_dir, 'generator_epoch-{epoch:04d}.weights.h5')
checkpoint_path_disc = os.path.join(checkpoint_dir, 'discriminator_epoch-{epoch:04d}.weights.h5')

n_epochs = 1000
batch_size = X_train_reco.shape[0]//100
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reco, X_train_truth, Y_train)).batch(batch_size)

m = 2  # Number of generator updates
n = 1  # Number of discriminator updates

performance_metric = np.empty((n_epochs, n_variables))
data_set_1 = X_test_truth[Y_test == 1]
data_set_2 = X_test_truth[Y_test == 0]
baseline = np.array([wasserstein_distance(data_set_1[:, i], data_set_2[:, i]) for i in range(n_variables)])

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")
    discriminator_losses = []
    generator_losses = []

    for X_detector_batch, X_particle_batch, Y_batch in train_dataset:
        # Discriminator update loop
        for _ in range(n):
            model_generator.trainable = False
            W_batch = model_generator.predict(X_particle_batch, verbose=0)
            W_batch = tf.where(Y_batch == 1, 1.0, tf.squeeze(W_batch))
            d_loss = train_step_discriminator(X_detector_batch, Y_batch, W_batch)
            discriminator_losses.append(d_loss.numpy())

        # Generator update loop
        for _ in range(m):
            alt_indices = tf.where(Y_batch == 0)
            X_particle_alt = tf.gather_nd(X_particle_batch, alt_indices)
            X_detector_alt = tf.gather_nd(X_detector_batch, alt_indices)
            Y_alt = tf.gather_nd(Y_batch, alt_indices)
            g_loss = train_step_gan(X_particle_alt, X_detector_alt, Y_alt)
            generator_losses.append(g_loss.numpy())

    # Log and printing specs about the model
    avg_d_loss = np.mean(discriminator_losses[-n*len(train_dataset):])
    avg_g_loss = np.mean(generator_losses[-m*len(train_dataset):])
    gen_loss_avg.append(avg_g_loss)
    disc_loss_avg.append(avg_d_loss)   
    print(f"Epoch {epoch+1} completed. Discriminator Loss: {avg_d_loss}, Generator Loss: {avg_g_loss}")
    if (epoch + 1) % 10 == 0:
        gen_checkpoint_path = checkpoint_path_gen.format(epoch=epoch + 1)
        disc_checkpoint_path = checkpoint_path_disc.format(epoch=epoch + 1)
        model_generator.save_weights(gen_checkpoint_path)
        model_discriminator.save_weights(disc_checkpoint_path)
        print(f'Saved generator and discriminator at epoch {epoch + 1}')
    weights = model_generator.predict(X_test_truth[Y_test == 0], verbose=0).flatten()
    performance_metric[epoch, :] = [wasserstein_distance(data_set_1[:, i], data_set_2[:, i], 
                                                          u_weights=None, v_weights=weights) 
                                    for i in range(n_variables)]

np.savez_compressed('training_metrics_compressed.npz', disc_loss_avg=disc_loss_avg, gen_loss_avg=gen_loss_avg, baseline=baseline, performance_metric=performance_metric)

