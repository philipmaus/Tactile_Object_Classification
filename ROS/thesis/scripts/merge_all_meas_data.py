#! /usr/bin/env python

import os
from os import listdir
from os.path import isfile, join
import csv
import numpy as np
import math


########## Functions #############

def sort_files_by_date():
	global files
	buf = np.zeros(len(files))
	indic = '_'
	for i in range(0,len(files)):
		buf[i] = files[i][files[i].index(indic)+1:-4]	# extract date info from file name
	#print(buf)
	buf, files = zip(*sorted(zip(buf, files)))			# sort buf from low to high and files accordingly
	#print(buf)
	print('Files sorted chronologically')

def read_csv_file(i):
	data_buf = []
	len_0 = 0
	filepath = mypath + files[i]
	with open(filepath, mode='r') as csv_file:
		csv_reader = csv.reader(csv_file)
		j = 0
		for row in csv_reader:
			if j == 0:
				data_buf = row
				len_0 = len(row)
			elif j == 1:
				data_buf = np.append(data_buf,np.zeros(len(row)-len_0))		# bring first row to same length as all others
				data_buf = np.vstack((data_buf, row))
			else:
				data_buf = np.vstack((data_buf, row))
			j += 1
		data_buf[0][2] = i # set measurement sequence number
		#print(data_buf)
		#print(data_buf.shape)
	return data_buf

def compute_eucl_force_norm(data_buf):
	data_only = data_buf[2:,:]
	header = data_buf[0,:]
	descr = data_buf[1,:]
	#print(data_only)
	eucl_1 = np.zeros((data_only.shape[0]))
	eucl_2 = np.zeros((data_only.shape[0]))
	eucl_3 = np.zeros((data_only.shape[0]))
	for i in range(0,data_only.shape[0]):
		eucl_1[i] = str(math.sqrt(np.float(data_only[i,1])**2+np.float(data_only[i,2])**2+np.float(data_only[i,3])**2))
		eucl_2[i] = str(math.sqrt(np.float(data_only[i,4])**2+np.float(data_only[i,5])**2+np.float(data_only[i,6])**2))
		eucl_3[i] = str(math.sqrt(np.float(data_only[i,7])**2+np.float(data_only[i,8])**2+np.float(data_only[i,9])**2))
	eucl_1.shape = (50,1)
	eucl_2.shape = (50,1)
	eucl_3.shape = (50,1)
	data_buff = np.hstack((data_only[:,:4], eucl_1, data_only[:,4:7], eucl_2, data_only[:,7:10], eucl_3, data_only[:,10:]))
	descr = np.insert(descr,4,'eucl force 1')
	descr = np.insert(descr,8,'eucl force 2')
	descr = np.insert(descr,12,'eucl force 3')
	header = np.hstack((header, np.zeros((3))))
	data_results = np.vstack((header,descr,data_buff))
	return data_results

def merge_csv_files():
	global data
	data_buf = []
	data_eucl = []
	for i in range(0, len(files)):
		data_buf = read_csv_file(i)
		data_eucl = compute_eucl_force_norm(data_buf)
		if i == 0:
			data = data_eucl
		else:
			data = np.vstack((data,data_eucl))
	#print(data)
	#print(data.shape)

def save_in_file():
	os.chdir('../')   # set working directory
	print('Working directory: ' + os.getcwd())
	filename =  '000_real_life_data.csv'
	with open(filename, 'w') as f:
		thewriter = csv.writer(f)
		thewriter.writerow(['Philip Maus','Master Thesis','Bionics Engineering',\
							'Scuola Superiore SantAnna','July 2020','Number of samples',len(files),'0','0','0','0','0','0','0','0','0'])
		for i in range(0, data.shape[0]):
			thewriter.writerow(data[i,:])
		f.close()
	print("Data saved as " + filename)


################ Main #######################

mypath = '../data_archive/in-lab/all/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print(files)
data = []
data_buf = []

sort_files_by_date()
#print(files)
merge_csv_files()
save_in_file()