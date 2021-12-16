#Data
import projectIrocs as p_irocs
import cv2 as cv
import numpy as np
import math
import sys
import time

from sklearn.mixture import GaussianMixture

#Code
if __name__ == "__main__":

	if len(sys.argv) != 4:
		print(sys.argv)
		print("ERROR :( : syntax is correct ")
		print("Do write what inside brakets")
		print("Syntax example:")
		print(" python name.py 64[block size 64x64] 32[step] image.png[test image]")
	else:

		#params given by the tester
		block = int(sys.argv[1])
		step = int(sys.argv[2])
		image = cv.imread(sys.argv[3])

		if image is None:
			sys.exit("ERROR: Couldn't read the image")
		elif block % 16 != 0:
			sys.exit("ERROR: block number isn't correct, verify documentation")
		elif step % 8 != 0:
			sys.exit("ERROR: step number isn't correct, verify documentation")
		else :
			#time execution start
			time_inicial = time.time()

			#change color space
			lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
			rows,cols, dim = lab.shape

			#init gmms
			#n = 2 clean and dirty
			gmm= GaussianMixture(n_components=2, random_state=0, 
								covariance_type='spherical',init_params='kmeans', 
								warm_start = True, max_iter = 200, n_init = 5)

			#auxiliar vars
			#probability
			p=[[],[],[]]
			pp=[[],[],[]]
			proba = [[],[],[]]

			#mean
			X_means = [[],[],[]]
			#stander deviaton
			X_std = [[],[],[]]
			#gradient magnitude
			Mag = np.zeros((rows,cols,dim),np.uint16)

			#probability per channel
			pl = np.zeros((rows,cols),np.uint16)
			pb = np.zeros((rows,cols),np.uint16)
			pr = np.zeros((rows,cols),np.uint16)

			#dirty cluster
			cl = [1,1,1]

			#for each channel 
			for i in range(dim):
				#init gradient
				prop = p_irocs.Gradient(lab[:,:,i])
				#magnitude gradint calculation
				Mag[:,:,i]=prop.Cal_Gradient_Magnitude("sobel")
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
				pp[i] = gmm.predict(X_test)
			
				if(np.count_nonzero(pp[i] == 1)>np.count_nonzero(pp[i] == 0)):
					pp[i] = np.logical_not(pp[i]).astype(int)
					cl[i] = 0

			for j in range(len(p[0])):
				#dirty zones
				I,J = p_irocs.Indexs(rows,cols,block,j,block/step)
				pl[I:I+block,J:J+block] =p[0][j][cl[0]]*255
				pb[I:I+block,J:J+block] =p[1][j][cl[1]]*255
				pr[I:I+block,J:J+block] =p[2][j][cl[2]]*255

			#calcule of probability produt
			produtorio =  pl * pb * pr
			pprodutorio = pp[0]*pp[1]*pp[2]

			#dirty zone
			print("Dirty Analizing ...")
			dirty_predict = np.zeros((rows,cols,dim),np.uint8)
			dirty_proba =  np.zeros((rows,cols,dim),np.uint8)

			#criteria
			crit = 0.6
			vec_proba = np.zeros((1,len(pprodutorio)),np.uint8)
			for i in range(len(pprodutorio)):
				#dirty zones
				I,J = p_irocs.Indexs(rows,cols,block,i,block/step)
				#predict mask
				if pprodutorio[i] >0.5:
					dirty_predict[I:I+block,J:J+block,:] = 1
				if np.mean(produtorio[I:I+block,J:J+block])>=255*crit:
					dirty_proba[I:I+block,J:J+block,:] = 1

			print("--- %s segundos ---" % (time.time() - time_inicial))

			cv.imwrite('mask_probability.png', dirty_proba*255)
			cv.imwrite('mask_prevision.png', dirty_predict*255)
