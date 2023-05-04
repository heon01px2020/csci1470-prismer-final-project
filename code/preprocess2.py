import pickle
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import cv2
import pandas as pd


def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data3(file_path, first_class, second_class):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This 
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param second_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	"""
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file[b'data']
	labels = unpickled_file[b'labels']
	# TODO: Do the rest of preprocessing!
	if not isinstance(labels, np.ndarray):
		labels = np.array(labels)
	classes_indices = ((labels == first_class) != (labels == second_class)).nonzero()
	inputs = tf.reshape(inputs[classes_indices]/255, (-1, 3, 32 ,32))
	inputs = tf.transpose(inputs, perm=[0,2,3,1])
	# normalizing labels
	labels = (labels[classes_indices]-min(first_class, second_class)) / (abs(first_class-second_class))
	# one-hot encoding labels
	labels = tf.one_hot(labels, 2)
	return inputs, labels

def get_data(file_path):
	unpickled_file = unpickle(file_path)
	inputs = unpickled_file['data']
	labels = unpickled_file['labels']
	# TODO: Do the rest of preprocessing!
	if not isinstance(labels, np.ndarray):
		labels = np.array(labels)
	inputs = tf.reshape(inputs/255, (-1, 3, 64 ,64))
	inputs = tf.transpose(inputs, perm=[0,2,3,1])
	# normalizing labels
	labels = labels - 1
	# one-hot encoding labels
	labels = tf.one_hot(labels, 1000)
	return inputs, labels

def get_train_data(file_path, mapping):
	images = os.listdir(file_path)

	train_inputs = []
	train_labels = []

	for image in images:
		key = image.split("_")[0]
		if key == ".DS":
			continue
		im = os.path.join(file_path, image)
		img = cv2.imread(im)
		img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
		img = np.reshape(img, (1, 32, 32, 3))
		train_inputs.append(img)
		train_labels.append(mapping[key])

	inputs = np.array(train_inputs)
	labels = np.array(train_labels)

	inputs = tf.reshape(inputs/255, (-1, 3, 32 ,32))
	inputs = tf.transpose(inputs, perm=[0,2,3,1])
	# one-hot encoding labels
	labels = tf.one_hot(labels, 200)
	return inputs, labels

def get_val_data(image_path, txt_path, category_dict):
	images = os.listdir(image_path)
	inputs = []
	for image in images:
		im = os.path.join(image_path, image)
		if image == ".DS":
			continue
		img = cv2.imread(im)
		img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
		img = np.reshape(img, (1, 32, 32, 3))
		inputs.append(img)
	
	df = pd.read_csv(txt_path, header=None, usecols=[1], sep='\t')
	df = np.array(df)

    # turn labels into 1-200 categories 
	labels = []
	for label in df:
		labels.append(category_dict[label[0]])
	
    # one-hot encoding labels
	labels = tf.one_hot(labels, 200)
	inputs = np.array(inputs)
	inputs = tf.convert_to_tensor(inputs)
	return inputs, labels