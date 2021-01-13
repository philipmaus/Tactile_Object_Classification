# # Tactile object classification with sensorized Jaco arm on Doro robot

# #### This is a function create augmented data sets.

# #### You can freely modify this code for your own purpose. However, please cite this work when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Philip Maus (philiprmaus@gmail.com)
# 
# https://github.com/philipmaus/Tactile_Object_Classification_with_sensorized_Jaco_arm_on_Doro_robot


import sqlite3
import os
from os import listdir
from os.path import isfile, join
import csv
import math
import numpy as np
from numpy import array
from numpy.linalg import norm
from numpy import sqrt
from numpy import savetxt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd



##################################### Load Data and Preprocess functions ##############################################


# load data from datafile
def load_data():
    # load data
    global number_samples
    data = []
    data_buf = []
    labels = []
    os.chdir('../')   # set working directory
    print('Working directory: ' + os.getcwd())
    filename =  '000_training_data.csv'
    data_buf = np.array(pd.read_csv(filename, header=None))
    number_samples = int(data_buf[0][6])                    # number of samples in data files
    nMeas = int(((data_buf.shape[0]-1)/number_samples)-2)   # number of measurements per sample
    header = data_buf[0]
    data_with_header = np.reshape(data_buf[1:],(number_samples, int((data_buf.shape[0]-1)/number_samples), data_buf.shape[1]))
    data = np.array(data_with_header[:,2:,1:],dtype=float)
    for i in range(0,data_with_header.shape[0]):
        labels = np.append(labels, int(data_with_header[i,0,10]))
    print(data_with_header.shape)   #samples x timesteps x features
    print(labels.shape)
    return data_with_header, labels

def preprocess(data_with_header):
    data = np.array(data_with_header[:,2:,1:],dtype=float)
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

    data_with_header[:,2:,1:] = data
    return data_with_header


##################################### Data augmentation methods ##############################################


### 1. Jittering: applying different noise to each sample, random noise added to each single value

sigma_jitter = 0.05    # standard devitation (STD) of the noise

def DA_Jitter(X, sigma):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise


### 2. Scaling: applying constant noise to the entire samples, all time steps of one feature multiplied with same noise factor

sigma_scaling = 0.1     # STD of the zoom-in/out factor

def DA_Scaling(X, sigma):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise


### 3. Magnitude Warping: applying smoothly-varing noise to the entire samples, every feature multiplied with cubic spline function around 1

sigma_magn_warp = 0.05     # STD of the random knots for generating curves
knot_magn_warp = 4        # # of knots for the random curves (complexity of the curves)

def GenerateRandomCurves(X, sigma, knot):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0]) # # timesteps
    result = np.ones((X.shape[0],X.shape[1]))
    for i in range(0,X.shape[1]):
        cs = CubicSpline(xx[:,i], yy[:,i])
        result[:,i] = np.array([cs(x_range)])
    return result

def DA_MagWarp(X, sigma, knot):
    curves = GenerateRandomCurves(X, sigma, knot)
    X_new = X * curves
    return X_new


### 4. Time Warping

sigma_time_warp = 0.2         # STD of the random knots for generating curves
knot_time_warp = 4            # # of knots for the random curves (complexity of the curves)

def DistortTimesteps(X, sigma, knot):
    tt = GenerateRandomCurves(X, sigma, knot)     # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)          # Add intervals to make a cumulative graph
    for i in range(0,X.shape[1]):
        t_scale = (X.shape[0]-1)/tt_cum[-1,i]  # Make the last value to have X.shape[0]
        tt_cum[:,i] = tt_cum[:,i]*t_scale
    return tt_cum

def DA_TimeWarp(X, sigma, knot):
    tt_new = DistortTimesteps(X, sigma, knot)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(0,X.shape[1]):
        X_new[:,i] = np.interp(x_range, tt_new[:,i], X[:,i])    # interpolation
    return X_new


### 5. Cropping          # crops out certain subset of sample, rest is set to 0

nSample = 5      # # of subsamples (nSample <= X.shape[0])
chosenSample = 3    # integer value of chosen subsample (0 to nSample-1)

def DA_Cropping(X, nSample, chosenSample):
    X_new = np.zeros(X.shape)
    X_sub = X[int(X.shape[0]/nSample*chosenSample):int(X.shape[0]/nSample*(chosenSample+1)),:]
    for i in range(int(X.shape[0]/nSample*chosenSample),int(X.shape[0]/nSample*(chosenSample+1))):
        X_new[i,:] = X_new[i,:] + X_sub[i-int(X.shape[0]/nSample*chosenSample),:]
    return X_new


##################################### Data save function #################################################


def save_data(data, labels):
    os.chdir('Data augmentation/')   # set saving directory
    filename =  'datasets/'+ augm_type + '_samples_' + str(rep*2000) + '_augmented_data.csv'
    with open(filename, 'w') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(['Philip Maus','Master Thesis','Bionics Engineering',\
                            'Scuola Superiore SantAnna','July 2020','Number of samples',sample_number*(rep+1),\
                            '0','0','0','0','0','0','0','0','0'])
        for i in range(0, data.shape[0]):
            if (i%meas_number == 0):        # set seq number
                data[i,2] = int((i-1)/meas_number+1)
            thewriter.writerow(data[i,:])
        f.close()
    print("Data saved as " + filename + " in " + os.getcwd())


##################################### Augment data function ##############################################


def augment_dataset(rep):     # creates #rep copies of each sample in dataset data with chosen augmentation method
    print('Data augmentation started')
    print('Type: ' + augm_type)
    print('Number of augmented measurements per sample: ' + str(rep))
    #load data
    data_with_header, labels = load_data()
    #preprocess data
    data_with_header = preprocess(data_with_header)
    global meas_number, sample_number
    sample_number = data_with_header.shape[0]
    meas_number = data_with_header.shape[1]
    feature_number = data_with_header.shape[2]
    #augment data
    header_buf = []
    data_augm = []
    for sample in range(0,data_with_header.shape[0]):
        #save header for later
        header_buf = data_with_header[sample,0:2,:]
        seq_buf = data_with_header[sample,2:,0]
        data = np.array(data_with_header[sample,2:,1:],dtype=float)
        #real data augmentation
        for i in range(0,rep):
            data_new = []

            if augm_type == 'jitter':
                data_new = DA_Jitter(data, sigma_jitter)
            if augm_type == 'scaling':
                data_new = DA_Scaling(data, sigma_scaling)
            if augm_type == 'magn_warp':
                data_new = DA_MagWarp(data, sigma_magn_warp, knot_magn_warp)
            if augm_type == 'time_warp':
                data_new = DA_TimeWarp(data, sigma_time_warp, knot_time_warp)
            if augm_type == 'crop':
                data_new = DA_Cropping(data, rep, i)
            # combinations 2-way
            if augm_type == 'jitter_magn_warp_added':
                data_buf = DA_Jitter(data, sigma_jitter)
                data_new = DA_MagWarp(data_buf, sigma_magn_warp, knot_magn_warp)
            if augm_type == 'jitter_time_warp_added':
                data_buf = DA_Jitter(data, sigma_jitter)
                data_new = DA_TimeWarp(data_buf, sigma_time_warp, knot_time_warp)
            if augm_type == 'jitter_crop_added':
                data_buf = DA_Jitter(data, sigma_jitter)
                data_new = DA_Cropping(data_buf, rep, i)
            if augm_type == 'scaling_time_warp_added':
                data_buf = DA_Scaling(data, sigma_scaling)
                data_new = DA_TimeWarp(data_buf, sigma_time_warp, knot_time_warp)
            if augm_type == 'scaling_crop_added':
                data_buf = DA_Scaling(data, sigma_scaling)
                data_new = DA_Cropping(data_buf, rep, i)
            if augm_type == 'magn_warp_time_warp_added':
                data_buf = DA_MagWarp(data, sigma_magn_warp, knot_magn_warp)
                data_new = DA_TimeWarp(data_buf, sigma_time_warp, knot_time_warp)
            if augm_type == 'magn_warp_crop_added':
                data_buf = DA_MagWarp(data, sigma_magn_warp, knot_magn_warp)
                data_new = DA_Cropping(data_buf, rep, i)
            # combinations 3-way
            if augm_type == 'jitter_time_warp_crop_added':
                data_buf = DA_Jitter(data, sigma_jitter)
                data_buf2 = DA_TimeWarp(data_buf, sigma_time_warp, knot_time_warp)
                data_new = DA_Cropping(data_buf2, rep, i)
            if augm_type == 'magn_warp_time_warp_crop_added':
                data_buf = DA_MagWarp(data, sigma_magn_warp, knot_magn_warp)
                data_buf2 = DA_TimeWarp(data_buf, sigma_time_warp, knot_time_warp)
                data_new = DA_Cropping(data_buf2, rep, i)
            if augm_type == 'magn_warp_time_warp_crop_added':
                data_buf = DA_Scaling(data, sigma_scaling)
                data_buf2 = DA_TimeWarp(data_buf, sigma_time_warp, knot_time_warp)
                data_new = DA_Cropping(data_buf2, rep, i)

            label_new = labels[sample]
            labels = np.append(labels, label_new)
            #append header adn sequence counter to newly augmented data
            data_new_with_header = np.insert(data_new, 0, seq_buf, axis=1)
            data_new_with_header = np.append(header_buf, data_new_with_header, axis=0)
            #append new augmented sample to former ones
            data_augm = np.append(data_augm, data_new_with_header)

        if sample%100==0:
            print(sample)

    data_augm = np.reshape(data_augm,(int(data_augm.shape[0]/(meas_number*feature_number)),meas_number,feature_number)) 
    buf = np.append(data_with_header, data_augm)
    data_result = np.reshape(buf,(int(buf.shape[0]/feature_number),feature_number))
    print(data_result.shape)
    save_data(data_result, labels)



##################################### Main ##############################################

if __name__ == '__main__':
    # define number of artifical samples for each sample in original data set
    rep = 4
    # options for augm_type: 'jitter','scaling','magn_warp','time_warp','crop',
    # 'jitter_magn_warp_added','jitter_time_warp_added','jitter_crop_added',
    # 'scaling_time_warp_added','scaling_crop_added','magn_warp_time_warp_added','magn_warp_crop_added'
    # 'jitter_time_warp_crop_added','magn_warp_time_warp_crop_added','magn_warp_time_warp_crop_added'
    augm_type = 'jitter'
    augment_dataset(rep)
    augm_type = 'magn_warp'        
    augment_dataset(rep)