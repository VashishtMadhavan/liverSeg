import os,glob
import caffe
import numpy as np
import cv2
import sys
import random

#TODO: add data augmentation
class MedicalDataLayer(caffe.Layer):
	"""
	Loading image, label from the medical data files
	"""	

	def setup(self, bottom, top):
		self.crop_size = int(params['crop_size'])
		self.data_file = params['data_file']
		self.batch_size = int(params['batch_size'])
		self.random = True
		self.mean = np.array((60.058, 60.058, 60.058)), dtype=np.float32)
		
		params = eval(self.param_str)
		self.idx = 0
	
		self.label_dir = '/x/vashishtm/data/medical/labels/'
		self.images = [x.rstrip() for x in open(self.data_file).readlines()]
		
		if self.random:
			random.shuffle(self.images)
		
		self.labels = [self.label_dir + x.split('/')[-1] for x in self.images]
		

	def reshape(self, bottom, top):
		self.data = self.load_image_batch()
		self.label = self.load_label_batch()
		
		top[0].reshape(*self.image.shape)
		top[1].reshape(*self.label.shape)

	def forward(self, bottom, top):
		top[0].data[...] = self.data
		top[1].data[...] = self.label
		
		self.idx += 2
		if self.idx > len(self.images):
			self.idx = self.idx % len(self.images)
			random.shuffle(self.images)

	def backward(self, top, propogate_down, bottom):
		pass

	def load_image_batch(self):
		first = self.idx % len(self.images)
		second = (self.idx + 1) % len(self.images)
		im = cv2.imread(self.images[first])
		im2 = cv2.imread(self.images[second])
	
		#TODO: add cropping
		im -= np.array(self.mean)
		im2 -= np.array(self.mean)
		return np.array([im,im2], dype=np.float32)
	
	def load_label_batch(self):
		first = self.idx % len(self.images)
                second = (self.idx + 1) % len(self.images)
		im = cv2.imread(self.labels[first])[:,:,0]
		im2 = cv2.imread(self.labels[second])[:,:,0]
		
		#TODO: add same cropping as image
		lab1 = im[np.newaxis, ...]
		lab2 = im2[np.newaxis, ...]
		return np.array([lab1,lab2], dtype=np.uint8)
