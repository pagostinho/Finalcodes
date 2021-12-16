import numpy as np
import math
import cv2 as cv
import sys

def Cal_properties(imageChannel,dim,step):
	rows, cols = imageChannel.shape
	img_splited = np.zeros((dim,dim),np.uint16)
	X=[]
	means = []
	std = []
	for i in range(0,rows-dim+1,step):
		for j in range(0,cols-dim+1,step):
			img_splited = imageChannel[i : i+dim , j : j+dim]
			m = np.mean(img_splited)
			s = np.std(img_splited)
			means.append(m)
			std.append(s)
			X.append([m,s])		

	return X, means, std

def Indexs(rows, cols, dim, num, step):
	n = 0
	for i in range(0,rows-dim+1,dim/step):
		for j in range(0,cols-dim+1,dim/step):
			if n == num :
				return i,j
			n+=1
class Gradient(object):
	'''
	
	Gradient of image calculation and Magnitude also
	
	Parameters
	---------
	imageChannel : np.zeros((self.rows,self.cols),np.uint8), an image

	'''

	def __init__(self, imageChannel):
		self.channel = imageChannel
		self.rows,self.cols= self.channel.shape
		self.grad_x = np.zeros((self.rows,self.cols),np.int16)
		self.grad_y = np.zeros((self.rows,self.cols),np.int16)
		self.Mag = np.zeros((self.rows,self.cols),np.int32)
	#Calculate gradient on y and x direction, respect. Uni dimensional image
	#Calculate magnitude gradient is sqrt(grad_x**2 + grad_y**2)
	def Cal_Gradient_Magnitude(self,grad):

		#Normal gradient
		if grad == "normal" :
			for i in range(self.rows):
				for j in range(self.cols):
					if j < self.cols-1:
						self.grad_x[i,j] = self.channel[i,j]-self.channel[i,j+1]
					if i < self.rows-1:
						self.grad_y[i,j] = self.channel[i,j]-self.channel[i+1,j]
					self.Mag[i,j] = math.sqrt(self.grad_x[i,j]**2 + self.grad_y[i,j]**2)

		#Sobel
		elif grad == "sobel":

			self.x = cv.Sobel(self.channel,cv.CV_64F,0,1,ksize=5)
			self.y = cv.Sobel(self.channel,cv.CV_64F,1,0,ksize=5)
			r,c = self.x.shape

			for i in range(r):
				for j in range(c):
					self.Mag[i,j] = math.sqrt(self.x[i,j]**2 + self.y[i,j]**2)
		#Laplacian
		elif grad == "laplacian":
			self.laplacian = cv.Laplacian(self.channel,cv.CV_64F)
			for i in range(self.rows):
				for j in range(self.cols):
					self.Mag[i,j] = math.sqrt(self.laplacian[i,j]**2)
		else:
			sys.exit("ERROR: Name gradient isn't correct")


		return self.Mag
	def get_dimensions(self):
		return (self.cols_channel,self.rows_channel)

def Cal_Mag(grad_x,grad_y,rows,cols):
		Mg = np.zeros((rows,cols),np.uint16)
		for i in range(rows):
			for j in range(cols):
				Mg[i,j] = abs(math.sqrt(math.pow(grad_x[i,j],2) + math.pow(grad_y[i,j],2))-255)

		return Mg

def classification(imageChannel,dim,step):
	channel = imageChannel
	rows,cols= channel.shape
	block = np.zeros((dim,dim),np.uint16)

	X=[]
	for i in range(0,rows-dim+1,step):
		for j in range(0,cols-dim+1,step):
			block = imageChannel[i : i+dim , j : j+dim]
			m = np.mean(block)

			if m!=0:
				X.append(1) #dirty class
			else:
				X.append(0) #clean class

	return X

def distancia_euclidiana(p,q):
	dim, dist = len(p), 0
	for i in range(dim-1):
		dist += (p[i]-q[i])**2
	return math.sqrt(dist)	

#knn for 2 classes
def KNN(train_in,train_out,newBlock,K):
	distances, dim = {}, len(train_in)
	#Calculate euclidian distnace for newBlock
	for i in range(dim):
		distances[i]= distancia_euclidiana(train_in[i], newBlock)

	#obteain the index of the nearest neighbors
	k_neighborhood = sorted(distances, key = distances.get)[:K]

	classe1, classe2 = 0,0

	for index in k_neighborhood:
		if train_out[index] == 1:
			classe1 += 1
		else:
			classe2 += 1

	if classe1 > classe2 :
		return 1
	else:
		return 2
