#Zona de dados
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

#Zona de Codigo
if __name__ == "__main__":

	if len(sys.argv) != 4:
		print(sys.argv)
		print("ERROR :( : syntax is correct ")
		print("Do write what inside brakets")
		print("Syntax example:")
		print(" python name.py 64[block size 64x64] 32[step] sobel[gradtype]")
	else:	
		
		#path to images
		img_path = glob.glob("/home/irocs/Desktop/IROCS/GMM/images/floor9/*.png")
		gt_path = glob.glob("/home/irocs/Desktop/IROCS/GMM/groundtruth/floor9/*.png")
		if img_path == 0:
			sys.exit("ERROR: Image path isn't correct, need to verify on script")
		elif int(sys.argv[1]) % 16 != 0:
			sys.exit("ERROR: block number isn't correct, verify documentation")
		elif int(sys.argv[2]) % 8 != 0:
			sys.exit("ERROR: step number isn't correct, verify documentation")
		else :


			#params given by the tester
			block = int(sys.argv[1])
			step = int(sys.argv[2])
			grad = sys.argv[3]

			count = 0
			accuracy_pred = 0
			accuracy_prob = 0
			time_final = 0
			#loop for all images
			for path, path_gt in zip(img_path, gt_path):
				count = count +1
				#time execution start
				time_inicial = time.time()

				#read image
				image = cv.imread(path)

				#image validation
				if image is None:
					sys.exit("ERROR: Couldn't read the image")

				#change color space
				lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
				rows,cols, dim = lab.shape

				#init gmms
				#n = 1 dirty
				#n = 2 clean and dirty
				gmm= GaussianMixture(n_components=2, random_state=0, 
									covariance_type='spherical',init_params='kmeans', 
									warm_start = True, max_iter = 200, n_init = 5)

				#auxiliar vars
				#probability
				p=[[],[],[]]
				pp=[[],[],[]]
				proba = [[],[],[]]

				Vec_inst = []
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

					time_inicial = time.time()				
					pp[i] = gmm.predict(X_test)
					time_final = time_final + time.time() - time_inicial					
					
					#print(np.count_nonzero(pp[i] == 1))
					#print(np.count_nonzero(pp[i] == 0))
					
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
				#produtorio =  pl * pb * pr
				#pprodutorio = pp[0]*pp[1]*pp[2]

				produtorio =  (pl * pb * pr)
				pprodutorio = (pp[0]*pp[1]*pp[2])
	

				#dirty zone
				print("Dirty Analizing ..." + str(count))
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
						vec_proba[0,i] = 1


				groundtruth = cv.imread(path_gt,cv.IMREAD_UNCHANGED)

				#Classification of classes with groudthruth
				Vec_inst = p_irocs.classification(groundtruth, block,step)
				accuracy_pred = accuracy_pred + float(np.count_nonzero((pprodutorio * Vec_inst) == 1))/float(Vec_inst.count(1))
				accuracy_prob = accuracy_prob + float(np.count_nonzero((vec_proba * Vec_inst) == 1))/float(Vec_inst.count(1))
				#print("--- %s segundos ---" % (time_final))
				
				#exectuion time
				

				'''
				fig, axs = plt.subplots(1,3)
				axs[0].imshow(Mag[:,:,0],cmap='jet')
				axs[0].set_title("LIGHTNESS")
				axs[1].imshow(Mag[:,:,1],cmap='jet')
				axs[1].set_title("Blue/Yellow")
				axs[2].imshow(Mag[:,:,2],cmap='jet')
				axs[2].set_title("Green/Red")

				plt.show()


				# Set up survey vectors
				xvec = np.linspace(0.001, 10.0, 100)
				yvec = np.linspace(0.001, 10.0, 100)

				# Set up survey matrices.  Design disk loading and gear ratio.
				x1, x2 = np.meshgrid(xvec, yvec)

				fig, axs = plt.subplots(1,3)
				#center = (max(X_means[0])+min(X_means[0]))/2, (max(X_std[0])+min(X_std[0]))/2
				center = (sum((X_means[0]/max(X_means[0])*10)) / len(X_means[0]), sum((X_std[0]/max(X_std[0])*10)) / len(X_std[0]))
				# Evaluate some stuff to plot
				obj = ((x1-center[0])**2)/8 + ((x2-center[1])**2)/2
				axs[0].scatter((X_means[0]/max(X_means[0]))*10, (X_std[0]/ max(X_std[0]))*10, c = 'r', s = 1)
				cntr = axs[0].contour(x1, x2, obj, [0.01,0.1,0.5,1,2,4],colors='black')
				axs[0].clabel(cntr, fmt="%2.1f", use_clabeltext=True)
				axs[0].set_xlabel('mean', fontsize=10)
				axs[0].set_ylabel('standard deviation', fontsize=10)
				axs[0].set_title("LIGHTNESS")

				#center=(max(X_means[1])+min(X_means[1]))/2, (max(X_std[1])+min(X_std[1]))/2
				center = (sum((X_means[1]/max(X_means[1])*10)) / len(X_means[1]), sum((X_std[1]/max(X_std[1])*10)) / len(X_std[1]))
				# Evaluate some stuff to plot
				obj = ((x1-center[0])**2)/4 + ((x2-center[1])**2)/2
				axs[1].scatter((X_means[1]/max(X_means[1]))*10, (X_std[1]/ max(X_std[1]))*10,  c = 'r', s = 1)
				cntr = axs[1].contour(x1, x2, obj, [0.01,0.1,0.5,1,2,4],colors='black')
				axs[1].clabel(cntr, fmt="%2.1f", use_clabeltext=True)
				axs[1].set_xlabel('mean', fontsize=10)
				axs[1].set_ylabel('standard deviation', fontsize=10)
				axs[1].set_title("Blue/Yellow")

				center = (sum((X_means[2]/max(X_means[2])*10)) / len(X_means[2]), sum((X_std[2]/max(X_std[2])*10)) / len(X_std[2]))
				# Evaluate some stuff to plot
				obj = ((x1-center[0])**2)/4 + ((x2-center[1])**2)/2
				ellipse = Ellipse(center, max(X_std[2]), max(X_means[2])/4, angle=30, alpha=0.1)
				axs[2].scatter((X_means[2]/max(X_means[2]))*10, (X_std[2]/ max(X_std[2]))*10,  c = 'r', s = 1)
				cntr = axs[2].contour(x1, x2, obj, [0.01,0.1,0.5,1,2,4],colors='black')
				axs[2].clabel(cntr, fmt="%2.1f", use_clabeltext=True)
				axs[2].set_xlabel('mean', fontsize=10)
				axs[2].set_xlabel('mean', fontsize=10)
				axs[2].set_ylabel('standard deviation', fontsize=10)
				axs[2].set_title("Green/Red")
				plt.show()

				fig, axs = plt.subplots(1,3)
				axs[0].imshow(pl,cmap = 'jet')
				axs[0].set_title("LIGHTNESS Prob")

				axs[1].imshow(pb,cmap = 'jet')
				axs[1].set_title("Blue/Yellow Prob")

				axs[2].imshow(pr,cmap = 'jet')
				axs[2].set_title("Green/Red Prob")

				plt.show()
				'''

				image = cv.imread(path)
				cv.imshow("Imagem",image)
				ap_mask = image*dirty_proba
				cv.imshow("Probabilidade",ap_mask)
				ap_mask = image*dirty_predict
				cv.imshow("Prevision",ap_mask)
				'''
				


				fig, axs = plt.subplots(1,3)
				axs[0].imshow(produtorio,cmap = 'jet')
				axs[0].set_title("probability mask")

				ap_mask = image*dirty_proba

				axs[1].imshow(ap_mask)
				axs[1].set_title("Aply mask")

				axs[2].imshow(image)
				axs[2].set_title("Original Image")


				plt.show()
				
				
				fig, axs = plt.subplots(1,3)
				axs[0].imshow(dirty_predict*255,cmap = 'jet')
				axs[0].set_title("Predict mask")

				ap_mask = image*dirty_predict

				axs[1].imshow(ap_mask)
				axs[1].set_title("Aply mask")

				axs[2].imshow(image)
				axs[2].set_title("Original Image")
				

				plt.show()
				'''

				cv.waitKey(0)
				cv.destroyAllWindows()
				
			print("Accuracy Prevision = " + str(float(accuracy_pred)/float(count)))
			print("Accuracy probability = " + str(float(accuracy_prob)/float(count)))
