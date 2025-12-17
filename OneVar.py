import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

#load the datasets
with np.load('q_data.npz') as data:
    x_true, x_true_alt, x_reco, x_reco_alt = data.values()
    
#define the loss functions
epsilon = K.epsilon()

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    weights_1 = K.sum(y_true * weights) + epsilon
    weights_0 = K.sum((1 - y_true) * weights) + epsilon
    
    # Clip the prediction value to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred)/weights_1 +
                         (1 - y_true) * K.log(1 - y_pred)/weights_0)
    return K.mean(t_loss)

def weighted_binary_crossentropy_GAN(y_true, y_pred):
    weights = tf.gather(y_pred, [1], axis=1) # event weights
    y_pred = tf.gather(y_pred, [0], axis=1) # actual y_pred for loss
    
    weights_1 = K.sum(y_true * weights) + epsilon
    weights_0 = K.sum((1 - y_true) * weights) + epsilon
        
    # Clip the prediction value to prevent NaN's and Inf's

    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = weights * ((1 - y_true) * K.log(1 - y_pred)/weights_0)

    return K.mean(t_loss)

#Create the generator model
mymodel_inputtest = Input(shape=(1,))
hidden_layer_1 = Dense(50, activation='relu', use_bias=False, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02))(mymodel_inputtest)
batch_norm_1 = BatchNormalization()(hidden_layer_1)
hidden_layer_2 = Dense(50, activation='relu', use_bias=False)(batch_norm_1)
batch_norm_2 = BatchNormalization()(hidden_layer_2)
hidden_layer_3 = Dense(50, activation='relu', use_bias=False)(batch_norm_2)
pre_outputs = Dense(1, activation='linear', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.01))(hidden_layer_3)
outputs = tf.exp(pre_outputs)
model_generator = Model(inputs=mymodel_inputtest, outputs=outputs)
                       
#Create the discriminator model, and compile it while model_discrimintor.trainable is still true so that model_discrimintor.train_on_batch will cause the discriminator to train on loss=weighted_binary_crossentropy, while the generator is constant

inputs_disc = Input((1, ))
hidden_layer_1_disc = Dense(50, activation='relu')(inputs_disc)
hidden_layer_2_disc = Dense(50, activation='relu')(hidden_layer_1_disc)
hidden_layer_3_disc = Dense(50, activation='relu')(hidden_layer_2_disc)
outputs_disc = Dense(1, activation='sigmoid')(hidden_layer_3_disc)
model_discrimintor = Model(inputs=inputs_disc, outputs=outputs_disc)
model_discrimintor.compile(loss=weighted_binary_crossentropy, optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

#Now create the GAN model, and now set model_discriminator.trainable = False, so that when we train this model on batch, it will update the generator on loss=weighted_binary_crossentropy while the discriminator is held constant
model_discrimintor.trainable = False
mymodel_gan = Input(shape=(1,))
gan_model = Model(inputs=mymodel_gan,outputs=concatenate([model_discrimintor(mymodel_gan),model_generator(mymodel_gan)]))
gan_model.compile(loss=weighted_binary_crossentropy_GAN, optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

#Define the datasets, and Train-Test split them
xvals_truth = np.concatenate([x_true_alt,x_true])
xvals_reco = np.concatenate([x_reco_alt,x_reco])
yvals = np.concatenate([np.zeros(len(x_true_alt)),np.ones(len(x_true))])
X_train_truth, X_test_truth, X_train_reco, X_test_reco, Y_train, Y_test = train_test_split(xvals_truth, xvals_reco, yvals)

#Train the neural networks
n_epochs = 5
n_batch = 128*100
n_batches = len(X_train_reco) // n_batch

for i in range(n_epochs):
    
    mypreds = model_generator.predict(X_test_truth,batch_size=1000)
    print("on epoch=",i,np.mean(mypreds),np.min(mypreds),np.max(mypreds))
    if np.isnan(np.mean(mypreds)):
        break
        
    for j in range(n_batches):
        X_batch = X_train_reco[j*n_batch:(j+1)*n_batch]
        Y_batch = Y_train[j*n_batch:(j+1)*n_batch]
        W_batch = model_generator(X_batch)
        W_batch = np.array(W_batch).flatten()

        W_batch[Y_batch==1] = 1  # Keep weights as is for real data (x_reco)
        Y_batch_2 = np.stack((Y_batch, W_batch), axis=1)

        model_discrimintor.train_on_batch(X_batch, Y_batch_2)  # Train discriminator on both x_reco and x_reco_alt
        gan_model.train_on_batch(X_batch[Y_batch==0],np.ones_like(Y_batch[Y_batch==0])) #Train generator only on x_reco_alt
        
#Now apply the generator to x_true_alt, to get the weights at truth level
weights = model_generator.predict(X_test_truth[Y_test==0])

#plot histograms to check if it worked
fig, ax = plt.subplots(figsize=(8, 6))

nbins = 30
bins = np.linspace(X_test_truth.min(),X_test_truth.max(),nbins)

_,_,_=plt.hist(X_test_truth[Y_test==1],bins=bins,alpha=0.5,label="truth",density=True, zorder=-1)
_,_,_=plt.hist(X_test_truth[Y_test==0],bins=bins,alpha=0.5,label="gen",density=True, zorder=0)
_,_,_=plt.hist(X_test_truth[Y_test==0],bins=bins,weights=weights,histtype="step",color="red",ls=":", lw=2,label="weighted gen",density=True, zorder=1)
plt.legend(fontsize=15)
plt.ylabel("Trials")
plt.xlabel("jet charge")
#plt.savefig("Unifold.pdf", bbox_inches='tight', transparent=True)
plt.show()