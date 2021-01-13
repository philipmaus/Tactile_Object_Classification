# # Tactile object classification with sensorized Jaco arm on Doro robot

# #### This is the function defining the NN models and training them.

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
from numpy import savetxt
import tensorflow as tf
import keras
from keras import layers
from keras import initializers
from keras import models
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
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
from keras import backend as K
import time



##################################### Data structure ############################################

#data structure
# 0: OptoForce 1 x
# 1: OptoForce 1 y
# 2: OptoForce 1 z
# 3: Euclidean norm OptoForce1 x, y, z
# 4: OptoForce 2 x
# 5: OptoForce 2 y
# 6: OptoForce 2 z
# 7: Euclidean norm OptoForce1 x, y, z
# 8: OptoForce 3 x
# 9: OptoForce 3 y
# 10: OptoForce 3 z
# 11: Euclidean norm OptoForce1 x, y, z
# 12: finger angle 1
# 13: finger angle 2
# 14: finger angle 3



######################################### Functions ############################################


# convert labels from string to numeric
def convert_labels(argument):
    switcher = {
    	'corn':				0,
    	'candle':			1,
    	'banana':			2,
    	'rubiks cube':		3,
    	'cotton pads':		4,
    	'knotted rope':		5,
    	'tissues':			6,
    	'etui':				7,
    	'green pepper':		8,
    	'stuffed animal':	9,
    	'pepper mill':		10,
    	'tennis ball':		11,
    	'tomato':			12,
    	'sponge':			13,
    	'grapes':			14,
    	'squash ball':		15,
    	'light bulb':		16,
    	'massage ball':		17,
    	'cucumber':			18,
    	'empty':			19
    }
    return switcher.get(argument,"Invalid value")


######################################### Load Data ############################################

# load data from datafile and bring in needed NN input shape
def load_data(filename):
	# load data
	global number_samples
	data = []
	data_buf = []
	labels = []
	#os.chdir('/')   # set working directory
	print('Working directory: ' + os.getcwd())
	print("Load data from " + filename)
	data_buf = np.array(pd.read_csv('../'+filename, header=None))
	number_samples = int(data_buf[0][6])					# number of samples in data files
	nMeas = int(((data_buf.shape[0]-1)/number_samples)-2)	# number of measurements per sample
	header = data_buf[0]
	data_with_header = np.reshape(data_buf[1:],(number_samples, int((data_buf.shape[0]-1)/number_samples), data_buf.shape[1]))
	data = np.array(data_with_header[:,2:,1:],dtype=float)
	for i in range(0,data_with_header.shape[0]):
		labels = np.append(labels, int(data_with_header[i,0,10]))

	# only do weight compensation and normalization for original dataset
	# augmented dataset have that already done
	if filename == '000_training_data.csv':
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
		for j in range(0,data.shape[2]):
			data[:,:,j] = data[:,:,j] - np.mean(data[:,:,j])
			data[:,:,j] = data[:,:,j] / np.std(data[:,:,j])

	# adapt data to NN type needed input
	if NN_type == "MLP":		# get last sample only
		data_MLP = []
		for i in range(0,len(data)):
			data_MLP.append(data[i][-1])
		data_MLP = np.array(data_MLP)
		return data_MLP, labels
	elif NN_type == "LSTM":
		return data, labels
	elif NN_type == "CNN" or NN_type == "CNNLSTM" or NN_type == "DCNN":	# [sample,timesteps,features,channel]
		data_CNN2D = data.reshape(data.shape[0],data.shape[1], data.shape[2],1)
		return data_CNN2D, labels
	elif NN_type == "ConvLSTM":	# [sample,timesteps,rows(1),columns(features),channel]
		data_ConvLSTM = data.reshape(data.shape[0],data.shape[1],1,data.shape[2],1)
		return data_ConvLSTM, labels
	else:
		print("Wrong NN type input!")
		exit()


######################################### Plots and save results ############################################

# plot train and validation loss and accuracy over epochs
def plot_history(history):
	# summarize history for accuracy
	figure = plt.figure
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	filename = 'results/graphs/' + NN_type + '_' + data_augm_type + '_accuracy_graph.png'
	plt.savefig(filename)
	#plt.show()
	plt.close()
	# summarize history for loss
	figure = plt.figure
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	filename = 'results/graphs/' +  NN_type + '_' + data_augm_type + '_loss_graph.png'
	plt.savefig(filename)
	#plt.show()
	plt.close()


# save best fold model and best fold history
def save_best_model_and_history(model, history):
	# save model
	filename = 'results/saved_models/best_' + NN_type + '_' + data_augm_type + '.h5'
	model.save(filename)
	print('Best ' + NN_type + ' model saved as ' + filename)
	# save history
	filename = 'results/saved_models/history_best_' + NN_type + data_augm_type + '.csv'
	with open(filename, 'w') as f:
		thewriter = csv.writer(f)
		thewriter.writerow(history.history['acc'])
		thewriter.writerow(history.history['val_acc'])
		thewriter.writerow(history.history['loss'])
		thewriter.writerow(history.history['val_loss'])
		f.close()
	print('Best ' + NN_type + ' model history saved as ' + filename)
	plot_history(history)


# plot and save average confusion matrix over all k crossfolds
def plot_confusion_matrix(labels_test, predict_classes):
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
	filename = 'results/graphs/' + NN_type + '_' + data_augm_type + '_confusion_matrix.png'
	plt.savefig(filename)
	#plt.show()
	plt.close()


######################################### NN models ###############################################

def DCNN_model(data, labels, train, test, i):
	labels = to_categorical(labels,20)	#one-hot for 20 classes
	n_timesteps, n_features, n_outputs = data.shape[1], data.shape[2], labels.shape[1]
	DCNN = keras.Sequential()
	DCNN.add(Conv2D(input_shape=(n_timesteps,n_features,1),filters=31,kernel_size=(1,2),padding="same", activation="relu"))
	DCNN.add(BatchNormalization())
	DCNN.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
	DCNN.add(Conv2D(filters=128, kernel_size=(2,3), padding="same", activation="relu"))
	DCNN.add(BatchNormalization())
	DCNN.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
	DCNN.add(Conv2D(filters=57, kernel_size=(3,1), padding="same", activation="relu"))
	DCNN.add(BatchNormalization())
	DCNN.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"))
	DCNN.add(Flatten())
	DCNN.add(Dense(units=103,activation="relu"))
	DCNN.add(BatchNormalization())
	DCNN.add(Dense(units=138,activation="relu"))
	DCNN.add(BatchNormalization())
	DCNN.add(Dense(units=n_outputs, activation="softmax"))
	DCNN.summary()
	#build model
	DCNN.compile(optimizer=keras.optimizers.Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
	mc = ModelCheckpoint('saved_models/temp/DCNN_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	#train DCNN
	history = DCNN.fit(data[train], labels[train], validation_split=0.11, batch_size=128, epochs=300, verbose=0, callbacks=[es, mc])
	#load best DCNN model
	DCNN_best = load_model('saved_models/temp/DCNN_best.h5')
	DCNN_best.save('results/saved_models/DCNN_'+data_augm_type+str(i)+'.h5')
	#predict outcome for confusion matrix
	predict_classes = DCNN_best.predict_classes(data[test], verbose=0)
	# accuracy: (tp + tn) / (p + n)
	scores = (accuracy_score(labels[test].argmax(axis=1), predict_classes))*100
	print('Accuracy: %f' % scores)
	return scores, predict_classes, DCNN_best, history


def ConvLSTM_model(data, labels, train, test, i):
	labels = to_categorical(labels,20)	#one-hot for 20 classes
	n_timesteps, n_features, n_outputs = data.shape[1], data.shape[3], labels.shape[1]
	#NN layers
	ConvLSTM = keras.Sequential()
	ConvLSTM.add(layers.ConvLSTM2D(filters=86, kernel_size=(1,4), strides=(1,3), input_shape=(n_timesteps,1,n_features,1), kernel_initializer= initializers.he_normal()))#(time_steps, rows, cols, channels)
	ConvLSTM.add(layers.Dropout(0.490))
	ConvLSTM.add(layers.Flatten())
	ConvLSTM.add(layers.Dense(158, activation='relu', kernel_initializer= initializers.he_normal()))
	ConvLSTM.add(layers.Dense(n_outputs, activation='softmax'))
	#build ConvLSTM
	ConvLSTM.compile(optimizer=keras.optimizers.Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
	mc = ModelCheckpoint('saved_models/temp/ConvLSTM_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	#train ConvLSTM
	history = ConvLSTM.fit(data[train], labels[train], validation_split=0.11, batch_size=1024, epochs=300, verbose=0, callbacks=[es, mc])
	#print(ConvLSTM.summary())
	#load best ConLSTM model
	ConvLSTM_best = load_model('saved_models/temp/ConvLSTM_best.h5')
	ConvLSTM_best.save('results/saved_models/ConvLSTM_'+data_augm_type+str(i)+'.h5')
	#predict outcome for confusion matrix
	predict_classes = ConvLSTM_best.predict_classes(data[test], verbose=0)
	# accuracy: (tp + tn) / (p + n)
	scores = (accuracy_score(labels[test].argmax(axis=1), predict_classes))*100
	print('Accuracy: %f' % scores)
	return scores, predict_classes, ConvLSTM_best, history


def CNNLSTM_model(data, labels, train, test, i):
	labels = to_categorical(labels,20)	#one-hot for 20 classes
	n_timesteps, n_features, n_outputs = data.shape[1], data.shape[2], labels.shape[1]
	#NN layers
	CNNLSTM = keras.Sequential()
	CNNLSTM.add(layers.TimeDistributed(layers.Conv1D(filters=98,kernel_size=8,strides=3,activation='relu', kernel_initializer= initializers.he_normal(),input_shape=(n_timesteps,n_features,1))))
	CNNLSTM.add(layers.TimeDistributed(layers.Dropout(0.306)))
	CNNLSTM.add(layers.TimeDistributed(layers.Flatten()))
	CNNLSTM.add(layers.LSTM(34))
	CNNLSTM.add(layers.Dropout(0.345))
	CNNLSTM.add(layers.Dense(n_outputs, activation='softmax'))
	#build CNNLSTM
	CNNLSTM.compile(optimizer=keras.optimizers.Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
	mc = ModelCheckpoint('saved_models/temp/CNNLSTM_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	#train CNNLSTM
	history = CNNLSTM.fit(data[train], labels[train], validation_split=0.11, batch_size=1024, epochs=300, verbose=0, callbacks=[es, mc])
	#load best CNNLSTM model
	CNNLSTM_best = load_model('saved_models/temp/CNNLSTM_best.h5')
	CNNLSTM_best.save('results/saved_models/CNNLSTM_'+data_augm_type+str(i)+'.h5')
	#predict outcome for confusion matrix
	predict_classes = CNNLSTM_best.predict_classes(data[test], verbose=0)
	# accuracy: (tp + tn) / (p + n)
	scores = (accuracy_score(labels[test].argmax(axis=1), predict_classes))*100
	print('Accuracy: %f' % scores)
	return scores, predict_classes, CNNLSTM_best, history

	
def CNN_model(data, labels, train, test, i):
	labels = to_categorical(labels,20)	#one-hot for 20 classes
	n_timesteps, n_features, n_outputs = data.shape[1], data.shape[2], labels.shape[1]
	print(data.shape)
	#NN layers
	CNN = keras.Sequential()
	CNN.add(layers.Conv2D(filters=22, kernel_size=(4,3), strides=(2,2), activation='relu', kernel_initializer=initializers.he_normal(), input_shape=(n_timesteps,n_features,1)))#height, width, channels
	CNN.add(layers.BatchNormalization())
	CNN.add(layers.Flatten())
	CNN.add(layers.Dense(86, activation='relu', kernel_initializer= initializers.he_normal()))
	CNN.add(layers.Dense(n_outputs, activation='softmax'))  # Add a softmax layer with 16 output units
	#build CNN2D
	CNN.compile(optimizer=keras.optimizers.Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
	mc = ModelCheckpoint('saved_models/temp/CNN_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	#train CNN2D
	history = CNN.fit(data[train], labels[train], validation_split=0.11, batch_size=64, epochs=300, verbose=0, callbacks=[es, mc])
	#load best CNN2D model
	CNN_best = load_model('saved_models/temp/CNN_best.h5')
	CNN_best.save('results/saved_models/CNN_'+data_augm_type+str(i)+'.h5')
	#predict outcome for confusion matrix
	predict_classes = CNN_best.predict_classes(data[test], verbose=0)
	# accuracy: (tp + tn) / (p + n)
	scores = (accuracy_score(labels[test].argmax(axis=1), predict_classes))*100
	print('Accuracy: %f' % scores)
	return scores, predict_classes, CNN_best, history


def LSTM_model(data, labels, train, test, i):
	labels = to_categorical(labels,20)	#one-hot for 20 classes
	n_timesteps, n_features, n_outputs = data.shape[1], data.shape[2], labels.shape[1]
	#NN layers
	LSTM = keras.Sequential()
	LSTM.add(layers.LSTM(units=24, input_shape=(n_timesteps,n_features)))
	LSTM.add(layers.Dropout(0.166))
	LSTM.add(layers.Dense(n_outputs, activation='softmax'))  # Add a softmax layer with 20 output units
	#build LSTM
	LSTM.compile(optimizer=keras.optimizers.Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
	mc = ModelCheckpoint('saved_models/temp/LSTM_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	#train LSTM
	history = LSTM.fit(data[train], labels[train], validation_split=0.111, batch_size=256, epochs=300, verbose=0, callbacks=[es, mc])
	#load best LSTM model
	LSTM_best = load_model('saved_models/temp/LSTM_best.h5')
	LSTM_best.save('results/saved_models/LSTM_'+data_augm_type+str(i)+'.h5')
	#predict outcome for confusion matrix
	predict_classes = LSTM_best.predict_classes(data[test], verbose=0)
	# accuracy: (tp + tn) / (p + n)
	scores = (accuracy_score(labels[test].argmax(axis=1), predict_classes))*100
	print('Accuracy: %f' % scores)
	return scores, predict_classes, LSTM_best, history


def MLP_model(data, labels, train, test, i):
	labels = to_categorical(labels,20)	#one-hot for 20 classes
	n_features, n_outputs = data.shape[1], labels.shape[1]
	print(data.shape)
	#NN layers
	MLP = keras.Sequential()
	MLP.add(layers.Dense(845, input_dim=n_features, activation='relu', kernel_initializer= initializers.he_normal()))  # Adds a densely-connected layer with 644 units to the model:
	MLP.add(layers.Dropout(0.409))					# dropout layer with rate 0.62
	MLP.add(layers.Dense(159, activation='relu', kernel_initializer= initializers.he_normal()))
	MLP.add(layers.Dropout(0.163))
	MLP.add(layers.Dense(n_outputs, activation='softmax'))  # Add a softmax layer with 20 output units:
	#build MLP
	MLP.compile(optimizer=keras.optimizers.Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])
	# simple early stopping
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
	mc = ModelCheckpoint('saved_models/temp/MLP_best.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
	#train MLP
	history = MLP.fit(data[train], labels[train], validation_split=0.11, batch_size=1024, epochs=300, verbose=0, callbacks=[es, mc])
	#load best MLP model
	MLP_best = load_model('saved_models/temp/MLP_best.h5')
	MLP_best.save('results/saved_models/MLP_'+data_augm_type+str(i)+'.h5')
	#predict outcome for confusion matrix
	predict_classes = MLP_best.predict_classes(data[test], verbose=0)
	# accuracy: (tp + tn) / (p + n)
	scores = (accuracy_score(labels[test].argmax(axis=1), predict_classes))*100
	print('Accuracy: %f' % scores)
	return scores, predict_classes, MLP_best, history



######################################### Execute function ############################################

# run an experiment
def run_experiment(file):
	print("Neural network model type: " + NN_type)
	# load data from csv dataset file
	data, labels = load_data(file)
	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	# k-crossfold with 10 repetitions
	n_splits = 10
	kfold = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
	i = 0
	cvscores = []
	predict_cl_mat = np.ones((int(n_splits),int(number_samples/n_splits)))
	labels_predict = np.ones((int(n_splits),int(number_samples/n_splits)))
	for train, test in kfold.split(data, labels):
		print("k fold: " + str(i+1) + " of " + str(n_splits))
		if NN_type == "MLP":
			scores,predict_cl,model,history = MLP_model(data, labels, train, test, i+1)
		elif NN_type == "LSTM":
			scores,predict_cl,model,history = LSTM_model(data, labels, train, test, i+1)
		elif NN_type == "CNN":
			scores,predict_cl,model,history = CNN_model(data, labels, train, test, i+1)
		elif NN_type == "CNNLSTM":
			scores,predict_cl,model,history = CNNLSTM_model(data, labels, train, test, i+1)
		elif NN_type == "ConvLSTM":
			scores,predict_cl,model,history = ConvLSTM_model(data, labels, train, test, i+1)
		elif NN_type == "DCNN":
			scores,predict_cl,model,history = DCNN_model(data, labels, train, test, i+1)
		else:
			print("invalid NN type")
			exit()
		# collect test accuracy values
		cvscores.append(scores)
		# save best model and history
		if i == 0 or i == np.argmax(cvscores):
			model_best = model
			history_best = history
		# prepare confusion matrix values
		labels_predict[i,:] = labels[test]
		predict_cl_mat[i,:] = predict_cl
		i += 1
	savetxt('results/accuracies_' + NN_type + '_' + data_augm_type + '.csv', cvscores, delimiter=',')
	print("Total average accuracy: " + "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
	# save best model
	save_best_model_and_history(model_best, history_best)
	# plot average confusion matrix over all folds
	plot_confusion_matrix(labels_predict, predict_cl_mat)



######################################### Main ##############################################

if __name__ == '__main__':
	train_data_file =  '000_training_data.csv'		# define source file for data, original or augmented
	NN_type = "MLP" 					# options: MLP, LSTM, CNN, CNNLSTM, ConvLSTM, DCNN
	data_augm_type = 'original'			# define augm type, this will be used as naming for all result files
	run_experiment(train_data_file)