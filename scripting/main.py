import os
import tensorflow as tf
import numpy as np
from scipy.stats import wasserstein_distance

from utils.setup_data import *
from models.build_generator import *
from models.build_critic import *
from utils.checkpoint_manager import *
from config.model_config import gen_learning_rate, disc_learning_rate, n_epochs, n_gen_updates, n_critic_updates
from train.train_step import *

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    ds_train, ds_val, ds_test = load_dataset()
    n_variables = ds_train.element_spec[0].shape[1]
    model_generator = build_generator(input_shape=n_variables)
    model_critic = build_critic(input_shape=n_variables)
    
    lr_schedule = lambda initial_learning_rate : tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1e5,
        decay_rate=0.96,
        staircase=True)
    
    optimizer_gen = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(gen_learning_rate))
    optimizer_critic = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(disc_learning_rate))
    
    checkpoint_dir = './model_checkpoints'
    checkpoint, checkpoint_manager, checkpoint_prefix = create_checkpoint_manager(
        checkpoint_dir, model_generator, model_critic, optimizer_gen, optimizer_critic
    )
    
    # Load the latest checkpoint if available
    load_latest_checkpoint(checkpoint_manager)
    
    crit_loss_avg = []
    gen_loss_avg = []
    performance_metric = np.empty((n_epochs, n_variables))

    X_test_truth = []
    Y_test = []
    
    for X_particle_batch, X_detector_batch, Y_batch in ds_test:
        X_test_truth.append(X_particle_batch.numpy())
        Y_test.append(Y_batch.numpy())
    
    # Convert to numpy arrays
    X_test_truth = np.concatenate(X_test_truth, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)
    
    # Separate the data based on the Y_test values
    data_set_1 = X_test_truth[Y_test == 1]
    data_set_2 = X_test_truth[Y_test == 0]
    
    # Calculate the baseline
    baseline = np.array([wasserstein_distance(data_set_1[:, i], data_set_2[:, i]) for i in range(n_variables)])
    # Training loop
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        critic_losses = []
        generator_losses = []
        
        for X_detector_batch, X_particle_batch, Y_batch in ds_train:
            for _ in range(n_critic_updates):  # Critic update loop
                W_batch = model_generator.predict(X_particle_batch, verbose=0)
                W_batch = tf.where(Y_batch == 1, 1.0, tf.squeeze(W_batch))
                d_loss = train_step_critic(model_critic, model_generator, X_detector_batch, Y_batch, W_batch, optimizer_critic)
                critic_losses.append(d_loss.numpy())
    
            for _ in range(n_gen_updates):  # Generator update loop
                alt_indices = tf.where(Y_batch == 0)
                X_particle_alt = tf.gather_nd(X_particle_batch, alt_indices)
                X_detector_alt = tf.gather_nd(X_detector_batch, alt_indices)
                Y_alt = tf.gather_nd(Y_batch, alt_indices)
                g_loss = train_step_gan(model_critic, model_generator, X_particle_alt, X_detector_alt, Y_alt, optimizer_gen)
                generator_losses.append(g_loss.numpy())
    
        crit_loss_avg += [np.mean(critic_losses[-n_critic_updates * len(ds_train):])]
        gen_loss_avg += [np.mean(generator_losses[-n_gen_updates * len(ds_train):])]
    
        # Save checkpoint
        save_checkpoint(checkpoint_manager, epoch)

        weights = model_generator.predict(X_test_truth[Y_test == 0], verbose=0).flatten()
        performance_metric[epoch, :] = [ wasserstein_distance(data_set_1[:, i], data_set_2[:, i], 
                                                              u_weights=None, v_weights=weights) 
                                        for i in range(n_variables)
                                       ]
    np.savez('losses.npz', crit_loss_avg=crit_loss_avg, gen_loss_avg=gen_loss_avg)
    np.savez('performance_metrics.npz', performance_metric=performance_metric, baseline=baseline)
if __name__ == "__main__":
    main()
