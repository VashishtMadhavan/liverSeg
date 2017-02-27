import os,glob
import caffe
import numpy as np
import cv2
import sys
import random

class MedicalDataLayer(caffe.Layer):
	"""
	Loading image, label from the medical data files
	"""	

	def setup(self, bottom, top):
		params = eval(self.param_str)
		self.image_size = (512,512)
		self.crop_size = int(params['crop_size'])
		self.data_file = params['data_file']
		self.batch_size = int(params['batch_size'])
		self.random = True
		self.mean = np.array([60.058, 60.058, 60.058], dtype=np.float32)
		
		self.idx = 0
	
		self.label_dir = '/x/vashishtm/data/medical/labels/'
		self.images = [x.rstrip() for x in open(self.data_file).readlines()]
		
		if self.random:
			random.shuffle(self.images)
		
		self.labels = [self.label_dir + x.split('/')[-1] for x in self.images]
	

	def random_crop(self, size, crop_size):
        	"""Generate a random crop of size = size"""
       		W,H = size
        	w,h = crop_size,crop_size
        	xmin,xmax,ymin,ymax = [0,w,0,h] # create a box of size w x h
        	xshift = np.random.randint(0, W-w-1)
        	yshift = np.random.randint(0, H-h-1)

       		xmin = xmin + xshift
        	xmax = xmax + xshift
        	ymin = ymin + yshift
        	ymax = ymax + yshift
        	return xmin,xmax,ymin,ymax

	def reshape(self, bottom, top):
		self.xmin,self.xmax,self.ymin, self.ymax = self.random_crop(self.image_size, self.crop_size)
		self.image = self.load_image_batch()
		self.label = self.load_label_batch()
		
		top[0].reshape(*self.image.shape)
		top[1].reshape(*self.label.shape)

	def forward(self, bottom, top):
		top[0].data[...] = self.data
		top[1].data[...] = self.label
		
		self.idx += self.batch_size
		if self.idx > len(self.images):
			self.idx = self.idx % len(self.images)
			random.shuffle(self.images)

	def backward(self, top, propogate_down, bottom):
		pass

	def load_image_batch(self):
		image_batch = []
		for i in range(self.batch_size):
			index = (self.idx + i) % len(self.images)
			im = np.array(cv2.imread(self.images[index]), dtype=np.float32)
			im -= np.array(self.mean)
			im = im[self.xmin:self.xmax,self.ymin:self.ymax]
			im = im.transpose((2,0,1))
			image_batch.append(im)
		return np.array(image_batch, dtype=np.float32)
	
	def load_label_batch(self):
		label_batch = []
		for i in range(self.batch_size):
			index = (self.idx + i) % len(self.images)
			im = np.array(cv2.imread(self.labels[index])[:,:,0],dtype=np.uint8)
			im = im[self.xmin:self.xmax, self.ymin:self.ymax]
			im = im[np.newaxis, ...]
			label_batch.append(im)
		return np.array(label_batch, dtype=np.uint8)
