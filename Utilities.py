import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
#import networkx as nx
import copy


import tensorflow as tf

from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow.keras as k
from sklearn.utils import shuffle

import tensorflow.keras.backend as K

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import linear_model
from collections import defaultdict
import random


from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler, PowerTransformer




def Get_D(Data):
    k = Data.corr()
    k.dropna(axis=0, how='all', inplace = True)
    k.dropna(axis=1, how='all', inplace = True)

    D = k.values
    D[D < np.median(D)] = 0
    #D[D < .3] = 0
    #D[D >= .3] = 1
    D = (D + D.T)/2 
    diagnoal_mat = np.diag(D.sum(axis=1))
    D = np.linalg.inv(np.sqrt(diagnoal_mat)).dot(D).dot(np.linalg.inv(np.sqrt(diagnoal_mat)))
    
    return D





# Define util functions

def mape_keras(y_true, y_pred, threshold=0.1):
    v = k.backend.clip(k.backend.abs(y_true), threshold, None)
    diff = k.backend.abs((y_true - y_pred) / v)
    return 100.0 * k.backend.mean(diff, axis=-1)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def mape(y_true, y_pred, threshold=0.1):
    v = np.clip(np.abs(y_true), threshold, None)
    diff = np.abs((y_true - y_pred) / v)
    return 100.0 * np.mean(diff, axis=-1).mean()

def transform(idxs):
    return [idxs[:, i] for i in range(idxs.shape[1])]

def set_session(device_count=None, seed=0):
    gpu_options = tf.GPUOptions(allow_growth=True)
    if device_count is not None:
        config = tf.ConfigProto(
            gpu_options=gpu_options, 
            device_count=device_count
        )
    else:
        config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)
    k.backend.set_session(sess)

    np.random.seed(seed)
    tf.set_random_seed(seed)
    return sess
  
def get_metrics(model, x, y, batch_size=1024):
    yp = model.predict(x, batch_size=batch_size, verbose=1).flatten()
    return {
        "rmse": float(rmse(y, yp)), 
        "mape": float(mape(y, yp)), 
        "mae": float(mae(y, yp))
    }



from keras import backend as K
def reg_time(weight_matrix):
    T_tilda = K.dot(K.transpose(weight_matrix), weight_matrix)
    d_M = np.ones((T_tilda.shape))
    np.fill_diagonal(d_M, 0)
    return 0.1 * K.sum(K.abs(T_tilda * d_M))


def create_W_matrix(Data):
    # create the 0,1 matrix for missing and non missing values
    X = Data.values.copy()
    for i in range (0, X.shape[0]):
        for j in range (0, X.shape[1]):
            if np.isnan(X[i,j]) == True:
                X[i,j] = 0
            else:
                X[i,j] = 1
    return X


def create_ori_W_nan(Data):
    # create the 0,1 matrix for missing and non missing values
    X = Data.values.copy()
    for i in range (0, X.shape[0]):
        for j in range (0, X.shape[1]):
            if np.isnan(X[i,j]) == False:
                X[i,j] = 1
    return X

def ortho_reg(w):
    m = ((K.dot(K.transpose(w), w)) - np.eye(w.shape[1]))
    return 0.01 * K.sum(K.abs(m))



def firm_reg(w):
    m =  (K.mean(K.square(D_firm - K.dot(w, K.transpose(w)))))
    #print(m.shape)
    return 0.01 * m
    
         

def custom_loss_2(layer_1, layer_2, D1, D2, lamda_1, lamda_2):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        return ((K.mean(K.square(y_pred - y_true))) 
                + lamda_1*(K.mean((K.abs(K.dot(K.transpose(layer_1), layer_1) - k.eye()))))
               +lamda_2*(K.mean((K.square(D2 - K.dot(layer_2, K.transpose(layer_2)))))))
   
    # Return a function
    return loss


def custom_loss(layer, D, lamda):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        return ((K.mean(K.square(y_pred - y_true))) + lamda*(K.mean((K.square(D - K.dot(layer, K.transpose(layer)))))))
   
    # Return a function
    return loss


def create_NLTC(shape, rank, nc):
    inputs = [k.Input(shape=(1,), dtype="int32") for i in range(len(shape))]
    
    #""" 
    embeds = [
        k.layers.Embedding(output_dim=rank, input_dim=shape[i])(inputs[i])
        for i in range(len(shape))
    ]
    
    #""" 
    """
    embeds = []
    for j in range(len(shape)):
        #if j == 0:
            #embeds.append(k.layers.Embedding(output_dim=rank, input_dim=shape[j], embeddings_regularizer = ortho_reg)(inputs[j]))   
        
        if j == 1:
        #elif j == 1:
            embeds.append(k.layers.Embedding(output_dim=rank, input_dim=shape[j], embeddings_regularizer = firm_reg)(inputs[j]))
    
        else:
            embeds.append(k.layers.Embedding(output_dim=rank, input_dim=shape[j])(inputs[j]))
    
    """ 
    
    #p = embeds[0]
    #print(p)
    
    #model.layers[0].get_weights()[0]
    #print(len(embeds))
    x = k.layers.Concatenate(axis=1)(embeds)
    x = k.layers.Reshape(target_shape=(rank, len(shape), 1))(x)
    x = k.layers.Conv2D(
        nc, 
        kernel_size=(1, len(shape)), 
        activation="relu", 
        padding="valid"
    )(x)
    x = k.layers.Conv2D(
        nc, 
        kernel_size=(rank, 1), 
        activation="relu", 
        padding="valid"
    )(x)
    x = k.layers.Flatten()(x)
    x = k.layers.Dense(nc, activation="relu")(x)
    outputs = k.layers.Dense(1, activation="linear")(x)
    model = k.Model(inputs=inputs, outputs=outputs)

    return model


def per_miss(File):
    # Calculate the percentage of missing value in a data file
     return File.isnull().sum().sum()/(File.shape[0]*File.shape[1])

def MSE(original, imputed):
    # calculate Mean Squared Error
    return np.square(original-imputed).mean().mean()    

def MPE(y_true, y_pred, threshold=0.01):
    v = np.copy(y_true)
    np.place(v, v==0, threshold)
    #v = np.clip(np.abs(y_true), threshold, None)
    diff = np.abs((y_true - y_pred) / v)
    return np.mean(diff, axis=-1).mean()


def Creat_missing(File, Perc):
    X = File.copy()
    for col in X:
        vals_to_nan = X[col].dropna().sample(frac= Perc).index
        X.loc[vals_to_nan, col] = np.NaN
    return X


def R_squared(original, predicted):
    Differ = np.square(original-predicted)
    m = np.mean(original) 
    denom = np.square(original - m)
    R_sq = 1 - ((Differ.sum())/denom.sum())
    return R_sq


def Get_performance(Y_true, Y_pred): 
    print ("R2 : ", np.round(R_squared(Y_true, Y_pred), 4))
    print ("MSE",  np.round(MSE(Y_true, Y_pred), 4))
    print ("MPE", np.round(MPE(Y_true, Y_pred), 4))
    
    return (np.round(R_squared(Y_true, Y_pred), 4), np.round(MSE(Y_true, Y_pred), 4), np.round(MPE(Y_true, Y_pred), 4)) 
    

def regularized_loss(layer_1, layer_2, D2, lamda_1, lamda_2):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        return ((K.mean(K.square(y_pred - y_true))) 
                + lamda_1*(K.mean((K.abs(K.dot(K.transpose(layer_1), layer_1) - K.eye()))))
               +lamda_2*(K.mean((K.square(D2 - K.dot(layer_2, K.transpose(layer_2)))))))
   
    # Return a function
    return loss



def regularized_loss_1(layer, D, lamda):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        return ((K.mean(K.square(y_pred - y_true))) + lamda*(K.mean((K.square(D - K.dot(layer, K.transpose(layer)))))))
   
    # Return a function
    return loss
    

def plot_los(dic):
    plt.figure(figsize=(10,6))
    #plt.subplot(1, 2, 1)
    plt.plot(dic['loss'], label='Training loss')
    plt.plot(dic['val_loss'], label='Validation loss')
    plt.legend(frameon=False, prop={'size': 20})
    #plt.title('Train and Validation loss')
    plt.xlabel('epoch', fontsize=22)
    plt.ylabel('loss', fontsize=22)

    #plt.subplot(1, 2, 2)
    #plt.plot(dic['val_mae'])
    #plt.title('validation MAE')
    #plt.xlabel('epoch')
    #plt.ylabel('MAE')
    #plt.savefig('convergence_COSTCO.png', dpi=500)
    plt.show()