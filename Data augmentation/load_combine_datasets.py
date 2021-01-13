# # Tactile object classification with sensorized Jaco arm on Doro robot

# #### This is a function to merge existing augmented datasets.

# #### You can freely modify this code for your own purpose. However, please cite this work when you redistributed the code to others. Please contact me via email if you have any questions. Your contributions on the code are always welcome. Thank you.

# Philip Maus (philiprmaus@gmail.com)
# 
# https://github.com/philipmaus/Tactile_Object_Classification_with_sensorized_Jaco_arm_on_Doro_robot


import numpy as np
import os
import pandas as pd
import csv


# load data from datafile
def load_data(file):
	# load data
	global number_samples
	data = []
	data_buf = []
	labels = []
	print('Working directory: ' + os.getcwd())
	filename = 'datases/' + file
	print("Load data from " + filename)
	data_buf = np.array(pd.read_csv(filename, header=None))
	return data_buf


def save_dataset(data, new_name):
    filename =  'datases/'+ new_name
    with open(filename, 'w') as f:
        thewriter = csv.writer(f)
        for i in range(0, data.shape[0]):
            thewriter.writerow(data[i,:])
        f.close()
    print("Data saved as " + filename)


def combine_two_datasets(method_1, method_2):
	file_1 = method_1 + '_samples_4000_augmented_data.csv'
	file_2 = method_2 + '_samples_4000_augmented_data.csv'
	new_name = method_1+'_'+method_2+'_samples_4000_augmented_data.csv'
	# load first dataset
	data_1 = load_data(file_1)
	data_1_size = data_1[0,6]
	# load second dataset
	data_2 = load_data(file_2)
	data_2 = data_2[(2000*52+1):,:] #cut original data so it is not double
	for i in range(0,data_2.shape[0]):
		if i%52==0:
			data_2[i,2] = int(data_2[i,2]) + int(data_1_size) - 2000	# correct seq numbers for consecutive order
	# combine datasets
	data = np.append(data_1, data_2, axis=0)
	data[0,6] = int((data.shape[0]-1)/52)	#redefine total number of samples in dataset
	# save in csv file
	save_dataset(data, new_name)


def combine_three_datasets(method_1, method_2, method_3):
	file_1 = method_1 + '_samples_4000_augmented_data.csv'
	file_2 = method_2 + '_samples_4000_augmented_data.csv'
	file_3 = method_3 + '_samples_4000_augmented_data.csv'
	new_name = method_1+'_'+method_2+'_'+method_3+'_samples_4000_augmented_data.csv'
	# load first dataset
	data_1 = load_data(file_1)
	data_1_size = data_1[0,6]
	# load second dataset
	data_2 = load_data(file_2)
	data_2 = data_2[(2000*52+1):,:] #cut original data so it is not double
	for i in range(0,data_2.shape[0]):
		if i%52==0:
			data_2[i,2] = int(data_2[i,2]) -2000 + int(data_1_size)	# correct seq numbers for consecutive order
	# combine first and second dataset
	data_12 = np.append(data_1, data_2, axis=0)
	data_12_size = int((data_12.shape[0]-1)/52)
	# load third dataset
	data_3 = load_data(file_3)
	data_3 = data_3[(2000*52+1):,:] #cut original data so it is not double
	for i in range(0,data_3.shape[0]):
		if i%52==0:
			data_3[i,2] = int(data_3[i,2]) - 2000 + int(data_12_size)	# correct seq numbers for consecutive order
	# combine first two and third dataset
	data = np.append(data_12, data_3, axis=0)
	data[0,6] = int((data.shape[0]-1)/52) #redefine total number of samples in dataset
	# save in csv file
	save_dataset(data, new_name)



if __name__ == '__main__':
	# # example of use:

	# # combine two datasets to one
	# method_1 = 'jitter'
	# method_2 = 'crop'
	# combine_two_datasets(method_1, method_2)

	# combine three datasets to one
	method_1 = 'jitter'
	method_2 = 'magn_warp'
	method_3 = 'crop'
	combine_three_datasets(method_1, method_2, method_3)