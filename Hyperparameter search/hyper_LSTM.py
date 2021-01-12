# # Tactile object classification with sensorized Jaco arm on Doro robot

# #### This is the hyperparameter search file for the LSTM.

# #### You can freely modify this code for your own purpose. However, please cite this work when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Philip Maus (philiprmaus@gmail.com)
# 
# https://github.com/philipmaus/Tactile_Object_Classification_with_sensorized_Jaco_arm_on_Doro_robot


import math
import numpy as np
import csv
import os
from os import listdir
from os.path import isfile, join
from numpy import array
from numpy.linalg import norm
from numpy import sqrt
from numpy import loadtxt
import tensorflow
import keras
import tensorflow as tf
from keras import layers
from keras import initializers
from keras import models
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import random
from random import sample
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import pandas as pd
import hyperopt
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
import time
from keras import backend as K



# load data from datafile
def load_data():
    # load data
    global number_samples
    data = []
    data_buf = []
    labels = []
    #os.chdir('/')   # set working directory
    print('Working directory: ' + os.getcwd())
    filename =  '../000_training_data.csv'
    data_buf = np.array(pd.read_csv(filename, header=None))
    number_samples = int(data_buf[0][6])                    # number of samples in data files
    nMeas = int(((data_buf.shape[0]-1)/number_samples)-2)   # number of measurements per sample
    header = data_buf[0]
    data_with_header = np.reshape(data_buf[1:],(number_samples, int((data_buf.shape[0]-1)/number_samples), data_buf.shape[1]))
    data = np.array(data_with_header[:,2:,1:],dtype=float)
    for i in range(0,data_with_header.shape[0]):
        labels = np.append(labels, int(data_with_header[i,0,10]))
    # object weight compensation
    for i in range(0,data.shape[0]):
        # compensate object weight effect on finger 2
        weight_2 = 0
        if np.amax(data[i,0:3,6]) > 0.05:   # only compensate if initial measurement over limit value
            weight_2 = np.mean(data[i,0:3,6])   # get weight as average of first 3 force measurments
            for j in range(0,data.shape[1]):    # for each timestep
                data[i,j,6] -= weight_2 * math.cos((data[i,j,13]-20)/360*2*math.pi) # normal force compensation
                data[i,j,5] += weight_2 * math.sin((data[i,j,13]-20)/360*2*math.pi) # tangential force compensation
        # compensate object weight effect on finger 2
        weight_3 = 0
        if np.amax(data[i,0:3,10]) > 0.05:
            weight_3 = np.mean(data[i,0:3,10])
            for j in range(0,data.shape[1]):
                data[i,j,10] -= weight_3 * math.cos((data[i,j,13]-20)/360*2*math.pi)
                data[i,j,9] += weight_3 * math.sin((data[i,j,13]-20)/360*2*math.pi)
    # normalize each feature over all samples to mean value 0 and std 1
    for j in range(0,data.shape[2]):
        data[:,:,j] = data[:,:,j] - np.mean(data[:,:,j])
        data[:,:,j] = data[:,:,j] / np.std(data[:,:,j])
    # adapt data to NN type needed input
    return data, labels



space = {
    'unitsLSTM': hp.choice('unitsLSTM', np.arange(8, 64, dtype=int)),
    'dropout': hp.uniform('dropout', 0.001, 1),
    'batch': hp.choice('batch', [8, 16, 32, 64, 128, 256, 512, 1024]),
    }



def LSTM_model(param):
    n_timesteps, n_features, n_outputs = data_train.shape[1], data_train.shape[2], labels_train.shape[1]
    #NN layers
    LSTM = keras.Sequential()
    LSTM.add(layers.LSTM(units=param['unitsLSTM'], input_shape=(n_timesteps,n_features)))
    LSTM.add(layers.Dropout(param['dropout']))
    LSTM.add(layers.Dense(n_outputs, activation='softmax'))  # Add a softmax layer with 20 output units
    #build LSTM
    LSTM.compile(optimizer=keras.optimizers.Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
    mc = ModelCheckpoint('saved_models/LSTM_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    #train LSTM
    history = LSTM.fit(data_train, labels_train, validation_split=0.111, batch_size=param['batch'], epochs=100, verbose=0, callbacks=[es, mc])
    #load best MLP model
    LSTM_best = load_model('saved_models/LSTM_best.h5')
    #predict outcome for confusion matrix
    predict_classes = LSTM_best.predict_classes(data_test, verbose=0)
    # accuracy: (tp + tn) / (p + n)
    scores = (accuracy_score(labels_test.argmax(axis=1), predict_classes))*100
    # clear tf backend
    if K.backend() == 'tensorflow':
        K.clear_session()
    del n_features, n_outputs, LSTM, es, mc, history, LSTM_best, predict_classes
    global counter
    counter += 1
    print("counter", counter)
    return {'loss': scores*(-1), 'status': STATUS_OK}


if __name__ == '__main__':
    counter = 0
    # load data
    data, labels = load_data()
    labels = to_categorical(labels,20)  #one-hot for 20 classes
    # shuffle data and labels
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    # split in train and test
    data_train = data[:1800,:]
    labels_train = labels[:1800]
    data_test = data[1800:,:]
    labels_test = labels[1800:]
    del data, labels
    # set start time
    start_time = time.time()

    trials = Trials()
    best = fmin(LSTM_model, space, algo=tpe.suggest, trials=trials, max_evals=500, verbose=0)

    end_time = time.time()
    duration = end_time - start_time
    print(duration)

    # save determined hyperparameters
    print('Working directory: ' + os.getcwd())
    filename =  'hyper_LSTM.csv'
    with open(filename, 'w') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['Hyperparameter search','LSTM','duration',duration])
        for i in range(0,best_array.shape[0]):
            thewriter.writerow(best_array[i,:])
        thewriter.writerow(trials.trials)
        f.close()
    print("Data saved as " + filename)