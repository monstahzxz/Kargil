import tensorflow as tf
import numpy as np
import os
import math
import cv2
import dlib
import matplotlib.pyplot as plt


def model(X, Y):
	x = tf.placeholder(tf.float32,[None,dims,dims,3],name='x')
	y = tf.placeholder(tf.float32,[None,output_classes],name='y')
	w1 = tf.get_variable("w1",[fc_size1,fc_size2],initializer = tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [fc_size2], initializer = tf.zeros_initializer())
	w2 = tf.get_variable("w2",[fc_size2,fc_size3],initializer = tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [fc_size3], initializer = tf.zeros_initializer())
	w3 = tf.get_variable("w3",[fc_size3,fc_size4],initializer = tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [fc_size4], initializer = tf.zeros_initializer())
	w4 = tf.get_variable("w4",[fc_size4,fc_size5],initializer = tf.contrib.layers.xavier_initializer())
	b4 = tf.get_variable("b4", [fc_size5], initializer = tf.zeros_initializer())
	w5 = tf.get_variable("w5",[fc_size5,fc_size6],initializer = tf.contrib.layers.xavier_initializer())
	b5 = tf.get_variable("b5", [fc_size6], initializer = tf.zeros_initializer())
	w6 = tf.get_variable("w6",[fc_size6,fc_size7],initializer = tf.contrib.layers.xavier_initializer())
	b6 = tf.get_variable("b6", [fc_size7], initializer = tf.zeros_initializer())
	w7 = tf.get_variable("w7",[fc_size7,fc_size8],initializer = tf.contrib.layers.xavier_initializer())
	b7 = tf.get_variable("b7", [fc_size8], initializer = tf.zeros_initializer())
	w8 = tf.get_variable("w8",[fc_size8,fc_size9],initializer = tf.contrib.layers.xavier_initializer())
	b8 = tf.get_variable("b8", [fc_size9], initializer = tf.zeros_initializer())
	w9 = tf.get_variable("w9",[fc_size9,fc_size10],initializer = tf.contrib.layers.xavier_initializer())
	b9 = tf.get_variable("b9", [fc_size10], initializer = tf.zeros_initializer())

	x_norm = tf.layers.batch_normalization(x, training=True)
	conv1 = tf.layers.conv2d(x_norm,filters=conv1_filters,kernel_size=conv1_ksize,activation=tf.nn.relu,padding=conv1_padding,strides=conv1_strides,name='conv1')
	#print('conv1 shape = ' + str(conv1.shape))
	conv2 = tf.layers.conv2d(conv1,filters=conv2_filters,kernel_size=conv2_ksize,activation=tf.nn.relu,padding=conv2_padding,strides=conv2_strides,name='conv2')
	#print('conv2 shape = ' + str(conv2.shape))
	pool2 = tf.layers.max_pooling2d(conv2,pool_size=max_pool2_pool_size,strides=max_pool2_strides,padding=max_pool2_padding,name='pool2')
	#print('pool2 shape = ' + str(pool2.shape))
	pool2_flattened = tf.reshape(pool2,shape=[-1,pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])
	#print('pool2_flattened shape = ' + str(pool2_flattened.shape))
	#print('&&&&&&&&&&&&&&&&&&&&&&&&')
	z1 = tf.add(tf.matmul(pool2_flattened,w1),b1)
	a1 = tf.nn.relu(z1)
	z2 = tf.add(tf.matmul(a1,w2),b2)
	a2 = tf.nn.relu(z2)
	z3 = tf.add(tf.matmul(a2,w3),b3)
	a3 = tf.nn.relu(z3)
	z4 = tf.add(tf.matmul(a3,w4),b4)
	a4 = tf.nn.relu(z4)
	z5 = tf.add(tf.matmul(a4,w5),b5)
	a5 = tf.nn.relu(z5)
	z6 = tf.add(tf.matmul(a5,w6),b6)
	a6 = tf.nn.relu(z6)
	z7 = tf.add(tf.matmul(a6,w7),b7)
	a7 = tf.nn.relu(z7)
	z8 = tf.add(tf.matmul(a7,w8),b8)
	a8 = tf.nn.relu(z8)
	z9 = tf.add(tf.matmul(a8,w9),b9)

	#y_hat_softmax = tf.nn.softmax(z4)
	#cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat_softmax), [1]))
	cost = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z9,labels=y,name='loss'))
	optimiser = tf.train.AdamOptimizer().minimize(cost)
	init = tf.global_variables_initializer()
	sess = tf.Session()
	saver = tf.train.Saver()

	sess.run(init)

	for epoch in range(epochs):
		e_z9, _, e_cost = sess.run([z9,optimiser,cost],feed_dict = {x:X,y:Y})
		print(e_z9)
		print(Y)
		print('Cost after epoch ' + str(epoch) + ' = ' + str(e_cost))

	saver.save(sess,'D:/Facial Recognition/source/model/model.ckpt')



def getData(dataset_path):
	X = []
	Y = []

	for people in os.listdir(dataset_path):
		curr_no_of_images = 0
		curr_path = dataset_path + people + '/'
		for image in os.listdir(curr_path):
			curr_no_of_images += 1
			sample = cv2.imread(curr_path + image)
			X.append(sample)
			print(image + ' complete')
		temp = [0] * output_classes
		temp[ground_truth[people]] = 1
		Y += [temp] * curr_no_of_images
		#for i in range(curr_no_of_images):
		#	Y.append(ground_truth[people]) 

	#print(sample.shape)
	return X, Y		


if __name__ == '__main__':
	dims = 200
	epochs = 40
	dataset_path = 'new_images/'
	embed_dims = 128
	output_classes = 5
	#learning_rate = 0.00005

	ground_truth = {
		'Darin' : 0,
		'Maddy' : 1,
		'Sanju' : 2,
		'Sonu' : 3,
		'Vivek' : 4
	}

	conv1_shape = dims
	conv1_filters = 16
	conv1_ksize = 10
	conv1_strides = 1
	conv1_padding = 'same'

	conv2_shape = dims / 2
	conv2_filters = 16
	conv2_ksize = 10
	conv2_strides = 2
	conv2_padding = 'same'

	max_pool2_shape = 48
	max_pool2_pool_size = 5
	max_pool2_strides = 2
	max_pool2_padding = 'valid'

	fc_size1 = max_pool2_shape * max_pool2_shape * conv2_filters
	fc_size2 = embed_dims * 16
	fc_size3 = embed_dims * 8
	fc_size4 = embed_dims * 4
	fc_size5 = embed_dims * 4
	fc_size6 = embed_dims * 4
	fc_size7 = embed_dims * 2
	fc_size8 = embed_dims
	fc_size9 = embed_dims
	fc_size10 = output_classes
	print(str(fc_size1) +  ' ' + str(fc_size2) + ' ' + str(fc_size3) + ' ' + str(fc_size4) + ' ' + str(fc_size5) + ' ' + str(fc_size6) + ' ' + str(fc_size7) + ' ' + str(fc_size8))

	X, Y = getData(dataset_path)
	#print(Y)
	print('-------------------')
	print('Dataset prepared...')
	print('-------------------')

	model(X,Y)
	print('-------------------')
	print('Training complete..')
	print('-------------------')