import tensorflow as tf

#hyperparameters
gen_model_width = 150
gen_model_depth = 15
disc_model_width = 50
disc_model_depth = 4
dropout_rate = 0.2
gen_learning_rate = 4e-5
disc_learning_rate = 5e-6
penalty_lambda = 5.0
batch_size = 1400
n_epochs = 75
n_gen_updates = 3
n_critic_updates = 1

alpha_profile = 0

substructure_variables = ['w', 'q', 'm', 'tau1s', 'tau2s']
data_streams = ['_true', '_true_alt', '_reco', '_reco_alt']
n_variables = len(substructure_variables)

kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)