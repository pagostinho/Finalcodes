#Zona de dados
import projectIrocs as p_irocs
import cv2 as cv
import numpy as np
import math
import sys
import time
import matplotlib.pyplot as plt
import glob

from sklearn.mixture import GaussianMixture

#Zona de Codigo
if __name__ == "__main__":

	if len(sys.argv) != 5:
		print(sys.argv)
		print("ERROR :( : syntax is correct ")
		print("Do write what inside brakets")
		print("Syntax example:")
		print(" python name.py test_im.png 64[block size 64x64] 32[step] sobel[gradtype]")
	else:	
		
		#path to images
		image = cv.imread(sys.argv[1])
		if image is None:
			sys.exit("ERROR: Couldn't read the image")
		elif int(sys.argv[2]) % 16 != 0:
			sys.exit("ERROR: block number isn't correct, verify documentation")
		elif int(sys.argv[3]) % 16 != 0:
			sys.exit("ERROR: step number isn't correct, verify documentation")
		else :


			#params given by the tester
			block = int(sys.argv[2])
			step = int(sys.argv[3])
			grad = sys.argv[4]

			time_inicial = time.time()
			#read image
			#cv.imshow("Image", image)
			#image validation
			#change color space
			lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
			rows,cols, dim = lab.shape
			#init gmms
			#n = 1 dirty
			#n = 2 clean and dirty
			gmm= GaussianMixture(n_components=1, random_state=0, 
								covariance_type='spherical',init_params='kmeans', 
								warm_start = True, max_iter = 200, n_init = 5)
			#auxiliar vars
			#probability
			p=[[],[],[]]
			#mean
			X_means = [[],[],[]]
			#stander deviaton
			X_std = [[],[],[]]
			#gradient magnitude
			Mag = np.zeros((rows,cols,dim),np.uint16)
			#probability per channel
			po = np.zeros((rows,cols,dim),np.uint16)
		
			#for each channel 
			for i in range(dim):
				#init gradient
				prop = p_irocs.Gradient(lab[:,:,i])
				#magnitude gradint calculation
				Mag[:,:,i]=prop.Cal_Gradient_Magnitude(grad)
				#Calcule magnitude gradient propriets mean and std
				X_t, X_means[i], X_std[i] =p_irocs.Cal_properties(Mag[:,:,i],block,step)
				#conver te a list to array
				X_train = np.array(X_t,np.uint16)
				#test = train
				X_test = X_train
				#Fit Gaussian Mixture Model (Train) 
				#gmm = gmm.fit(X)
				gmm.fit(X_train)
				#calculate probability of floor (Test)
				p[i] = gmm.predict_proba(X_test)

			for j in range(len(p[0])):
				print(float(p[i][j][0]))
				#dirty zones
				I,J = p_irocs.Indexs(rows,cols,block,j,block/step)
				po[I:I+block,J:J+block,i] = p[i][j][0]*255
				

			print(po.shape)
			#calcule of probability produt
			produtorio = 1 - p[0]*p[1]*p[2]	
			#dirty zone
			print("Dirty Analizing ...")
			dirty = np.zeros((rows,cols,dim),np.uint8)
			#probabilty
			color_prob= np.zeros((rows,cols),np.uint16)

			criterio = 0.9
			for i in range(len(produtorio)):
				#dirty zones
				I,J = p_irocs.Indexs(rows,cols,block,i,block/step)
				#max 255 and min 0
				prob1 = produtorio[i][0]*255
				#prob2 = produtorio[i][1]*255
				#pp = 255 - (prob1+prob2)
				color_prob[I:I+block,J:J+block] = prob1
				#crit of dirty
				if prob1> (255*criterio):
					dirty[I:I+block,J:J+block,:] = 1

			#exectuion time
			print("--- %s segundos ---" % (time.time() - time_inicial))
			
			Titles = ["LIGHTNESS","Blue/Yellow","Green/Red"]
			fig, axs = plt.subplots(1,3)
			for i in range(3):
				axs[i].imshow(Mag[:,:,i],cmap='jet')
				axs[i].set_title(Titles[i])
			plt.show()


			# Set up survey vectors
			xvec = np.linspace(0.001, 10.0, 100)
			yvec = np.linspace(0.001, 10.0, 100)

			# Set up survey matrices.  Design disk loading and gear ratio.
			x1, x2 = np.meshgrid(xvec, yvec)

			fig, axs = plt.subplots(1,3)
			for i in range(3):
				center = (sum((X_means[i]/max(X_means[i])*10)) / len(X_means[i]), sum((X_std[i]/max(X_std[i])*10)) / len(X_std[i]))
				# Evaluate some stuff to plot
				obj = ((x1-center[0])**2)/4 + ((x2-center[1])**2)/2
				axs[i].scatter((X_means[i]/max(X_means[i]))*10, (X_std[i]/ max(X_std[i]))*10, c = 'r', s = 1)
				cntr = axs[i].contour(x1, x2, obj, [0.01,0.1,0.5,1,2,4],colors='black')
				axs[i].clabel(cntr, fmt="%2.1f", use_clabeltext=True)
				axs[i].set_xlabel('mean', fontsize=10)
				axs[i].set_ylabel('standard deviation', fontsize=10)
				axs[i].set_title(Titles[i])

			plt.show()

			fig, axs = plt.subplots(1,3)
			for i in range(3):
				axs[i].imshow(po[i],cmap = 'jet')
				axs[i].set_title(Titles[i])

			plt.show()


			fig, axs = plt.subplots(1,3)
			axs[0].imshow(color_prob,cmap = 'jet')
			axs[0].set_title("Prob cluster 1")

			ap_mask = image*dirty

			axs[1].imshow(ap_mask)
			axs[1].set_title("Aply mask")

			axs[2].imshow(image)
			axs[2].set_title("Original Image")

			plt.show()

			ap_mask = image*dirty
			cv.imshow("Mask", ap_mask)

			cv.waitKey(0)
			cv.destroyAllWindows()