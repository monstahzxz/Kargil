import tensorflow as tf
import numpy as np
import os
import cv2
import dlib
import matplotlib.pyplot as plt

curr_dir = ''#'/'.join(os.getcwd().split('\\'))
images_path = curr_dir + 'images'
dest_path = curr_dir + 'new_images/'

detector = dlib.get_frontal_face_detector()
dims = (200,200)


def sample_given_pic(sample_pic_path):
	print(sample_pic_path)
	img = cv2.imread(sample_pic_path)
	#cv2.imshow('sa',img)
	det = detector(img,1)
	d = det[0]

	img_cropped = img[d.top():d.bottom(),d.left():d.right()]
	img_resized = cv2.resize(img_cropped,dims)
	cv2.imwrite(dest_path + people + sample_pic_path.split('/')[-1],img_resized)
	print('Writing image complete')



if __name__ == '__main__':
	stop = 1
	for people in os.listdir(images_path):
		if people != 'Sonu' and stop == 1:
			continue
		stop = 0	
		curr_path = images_path + '/' + people + '/'
		for sample_pic in os.listdir(curr_path):
			sample_pic_path = curr_path + sample_pic
			sample_given_pic(sample_pic_path)