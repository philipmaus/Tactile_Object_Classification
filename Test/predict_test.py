# # Tactile object classification with sensorized Jaco arm on Doro robot

# #### This file predicts the success rate for real-life data by applying hem to the trained NN models.

# #### You can freely modify this code for your own purpose. However, please cite this work when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Philip Maus (philiprmaus@gmail.com)
# 
# https://github.com/philipmaus/Tactile_Object_Classification_with_sensorized_Jaco_arm_on_Doro_robot


import math
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import numpy as np
import csv
import os
from os import listdir
from os.path import isfile, join
from numpy import array
import tensorflow as tf
import keras
from keras.models import load_model
import random
from random import sample
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import pandas as pd
from collections import Counter


# load real-life data from datafile
def load_data():
	# load data
	global number_samples
	data = []
	data_buf = []
	labels = []
	filename =  '000_real_life_data.csv'
	data_buf = np.array(pd.read_csv(filename, header=None))

	number_samples = int(data_buf[0][6])					# number of samples in data files
	nMeas = int(((data_buf.shape[0]-1)/number_samples)-2)	# number of measurements per sample
	header = data_buf[0]
	data_with_header = np.reshape(data_buf[1:],(number_samples, int((data_buf.shape[0]-1)/number_samples), data_buf.shape[1]))
	data = np.array(data_with_header[:,2:,1:],dtype=float)
	for i in range(0,data_with_header.shape[0]):
		labels = np.append(labels, int(data_with_header[i,0,10]))
	print("\nData loaded: " + str(data.shape))
	# object weight compensation
	print("Object weight compensation")
	for i in range(0,data.shape[0]):
		# compensate object weight effect on finger 2
		weight_2 = 0
		if np.amax(data[i,0:3,6]) > 0.05:	# only compensate if initial measurement over limit value
			weight_2 = np.mean(data[i,0:3,6])	# get weight as average of first 3 force measurments
			for j in range(0,data.shape[1]):	# for each timestep
				data[i,j,6] -= weight_2 * math.cos((data[i,j,13]-20)/360*2*math.pi)	# normal force compensation
				data[i,j,5] += weight_2 * math.sin((data[i,j,13]-20)/360*2*math.pi)	# tangential force compensation
		# compensate object weight effect on finger 2
		weight_3 = 0
		if np.amax(data[i,0:3,10]) > 0.05:
			weight_3 = np.mean(data[i,0:3,10])
			for j in range(0,data.shape[1]):
				data[i,j,10] -= weight_3 * math.cos((data[i,j,13]-20)/360*2*math.pi)
				data[i,j,9] += weight_3 * math.sin((data[i,j,13]-20)/360*2*math.pi)

	# normalize each feature over all samples to mean value 0 and std 1
	print("Data normalization")
	mean_dataset = [-2.33293800e-01, 4.34488090e-01, -2.10292800e-02, 5.65528053e-01,\
					3.87262300e-02, 3.98179976e-02, 5.67078027e-01, 7.16047705e-01,\
					-3.02454720e-01, 2.86651377e-01, 7.50420239e-02, 4.68298448e-01,\
					3.31116600e+01, 3.35939650e+01, 3.35777650e+01]
	std_dataset = [0.39165433, 0.57594662, 0.21116683, 0.6734428, 0.267279, 0.26111209,\
					0.67091961, 0.72118393, 0.28212706, 0.27444994, 0.19451229, 0.39772967,\
					6.77925638, 7.47350691, 7.58818243]
	for j in range(0,data.shape[2]):
		data[:,:,j] = data[:,:,j] - mean_dataset[j]
		data[:,:,j] = data[:,:,j] / std_dataset[j]

	print("\nOverview over object occurrences:")
	count = Counter(labels)
	for i in range(0,20):
		print("Object " + str(i) + " occurrence: " + str(count[float(i)]))

	# adapt data shape to NN model input
	data = data.reshape(data.shape[0],data.shape[1], data.shape[2],1)

	return data, labels


def plot_confusion_matrix(labels_test, predict_classes, title):
	class_names = ['corn','candle','banana','rubiks cube','cotton pads','knotted rope','tissues','etui','green pepper','stuffed animal','pepper mill','tennis ball','tomato','sponge','grapes','squash ball','light bulb','massage ball','cucumber','empty']
	con_mat_sum = np.zeros((20,20))
	for i in range(0,labels_test.shape[0]):
		con_mat = confusion_matrix(labels_test[i,:], predict_classes[i,:])
		con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)	# normalize con mat
		con_mat_sum += con_mat_norm
	con_mat_sum /= labels_test.shape[0]
	con_mat_sum = np.around(con_mat_sum, decimals=2)
	figure = plt.figure(figsize=(10, 10))
	sns.heatmap(con_mat_sum, annot=True,cmap=plt.cm.Blues, xticklabels=class_names,yticklabels=class_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.title(title)
	filename = title + '_confusion_matrix.png'
	plt.savefig(filename)
	# plt.show()
	plt.close()



def predict():
	# load real-life data
	data, labels = load_data()
		# predict one model to get tensorflow messages out of the way
	filepath = 'trained_models/VGG16_scaling_time_warp1.h5'
	model = load_model(filepath)
	# predict real-life success rate for original dataset
	t = "VGG16"
	a = 'scaling_time_warp'
	filepath = ''
	predict_cl_mat = np.ones((10,data.shape[0]))
	labels_predict = np.ones((10,data.shape[0]))
	scores = []
	for i in range(0,10):
		filepath = 'trained_models/'+t+'_'+a+str(i+1)+'.h5'
		model = load_model(filepath)
		predict_classes = model.predict_classes(data, verbose=0)
		accuracy = (accuracy_score(labels, predict_classes))*100
		scores = np.append(scores,accuracy)
		labels_predict[i,:] = labels
		predict_cl_mat[i,:] = predict_classes
	print(t+' '+a+' dataset accuracy: ' + str(np.around(np.mean(scores),2))+' +- '+str(np.around(np.std(scores),2)))
	title = t + ' ' + a
	plot_confusion_matrix(labels_predict, predict_cl_mat,title)



if __name__ == '__main__':
	predict()