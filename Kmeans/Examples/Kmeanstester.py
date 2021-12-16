import projectIrocs as p_irocs
import cv2 as cv
import numpy as np
import math
import sys
import time
import matplotlib.pyplot as plt
import glob


from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

if __name__ == "__main__":

	
	img_path = glob.glob("/home/irocs/Desktop/IROCS/Project/food/*.png")

	for path in img_path:
		time_inicial = time.time()
		image = cv.imread(path)
		block = int(sys.argv[1])
		step = int(sys.argv[2])
		grad = sys.argv[3]
		if image is None:
			sys.exit("ERROR: Couldn't read the image")

		#img_splited = image[:dim,:dim]
		lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
		rows,cols, dim = lab.shape

		#Convolution
		kernel = np.ones((5,5),np.float32)/25
		lab = cv.filter2D(lab,-1,kernel)

		#init gmms
		#n = 2 clean and dirty
		
		p=[[],[],[]]
		X_means = [[],[],[]]
		X_std = [[],[],[]]
		Mag = np.zeros((rows,cols,dim),np.uint16)


		#for each channel 
		for i in range(dim):
			prop = p_irocs.Gradient(lab[:,:,i])
			Mag[:,:,i]=prop.Cal_Gradient_Magnitude(grad)
			X_t, X_means[i], X_std[i] =p_irocs.Cal_properties(Mag[:,:,i],block,step)
		
			Z = np.vstack(X_t)
			# convert to np.float32
			Z = np.float32(Z)

			# define criteria and apply kmeans()
			criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			ret,label,center=cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
			A = Z[label.ravel()==0]
			B = Z[label.ravel()==1]
			print(center)
			print(np.where(center[:,0]==min(center[:,0])))
			plt.scatter(A[:,0],A[:,1])
			plt.scatter(B[:,0],B[:,1],c = 'r')
			plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
			plt.xlabel('Mean'),plt.ylabel('STD')
			plt.show()


		
		print("Dirty Analizing ...")
		dirty = np.zeros((rows,cols,dim),np.uint8)
		

		print("--- %s segundos ---" % (time.time() - time_inicial))			
		


		'''
		fig, axs = plt.subplots(2,2)
		axs[0,0].imshow(Mag[:,:,0],cmap='jet')
		axs[0,0].set_title("LIGHTNESS")
		axs[0,1].imshow(Mag[:,:,1],cmap='jet')
		axs[0,1].set_title("Blue/Yellow")
		axs[1,0].imshow(Mag[:,:,2],cmap='jet')
		axs[1,0].set_title("Green/Red")

		plt.show()

		fig, axs = plt.subplots(2,2)
		#center = (max(X_means[0])+min(X_means[0]))/2, (max(X_std[0])+min(X_std[0]))/2
		center = (sum(X_means[0]) / len(X_means[0]), sum(X_std[0]) / len(X_std[0]))
		ellipse = Ellipse(center, max(X_std[0]), max(X_means[0])/8, angle=28, alpha=0.1)
		axs[0,0].scatter(X_means[0], X_std[0], c = 'r', s = 1)
		axs[0,0].add_artist(ellipse)
		axs[0,0].set_xlabel('mean', fontsize=10)
		axs[0,0].set_ylabel('standard deviation', fontsize=10)
		axs[0,0].set_title("LIGHTNESS")

		#center=(max(X_means[1])+min(X_means[1]))/2, (max(X_std[1])+min(X_std[1]))/2
		center = (sum(X_means[1]) / len(X_means[1]), sum(X_std[1]) / len(X_std[1]))
		ellipse = Ellipse(center, max(X_std[1])/2, max(X_means[1])/8, angle=28, alpha=0.1)
		axs[0,1].scatter(X_means[1], X_std[1],  c = 'r', s = 1)
		axs[0,1].add_artist(ellipse)
		axs[0,1].set_xlabel('mean', fontsize=10)
		axs[0,1].set_ylabel('standard deviation', fontsize=10)
		axs[0,1].set_title("Blue/Yellow")

		center = (sum(X_means[2]) / len(X_means[2]), sum(X_std[2]) / len(X_std[2]))
		ellipse = Ellipse(center, max(X_std[2])/1.5, max(X_means[2])/8, angle=28, alpha=0.1)
		axs[1,0].scatter(X_means[2], X_std[2],  c = 'r', s = 1)
		axs[1,0].add_artist(ellipse)
		axs[1,0].set_xlabel('mean', fontsize=10)
		axs[1,0].set_ylabel('standard deviation', fontsize=10)
		axs[1,0].set_title("Green/Red")
		plt.show()

		fig, axs = plt.subplots(1,3)
		axs[0].imshow(color_prob1,cmap='jet')
		axs[0].set_title("Probability 1")
		axs[1].imshow(color_prob2,cmap='jet')
		axs[1].set_title("Probability 2")
		axs[2].imshow(image)
		axs[2].set_title("Image")
		plt.show()

		ap_mask = image*dirty
		cv.imshow("Aplication Mask",ap_mask)
		'''

		cv.waitKey(0)
		cv.destroyAllWindows()	