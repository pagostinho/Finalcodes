#Data
import projectIrocs as p_irocs
import percepton as per
import cv2 as cv
import numpy as np
import sys
import time
import glob

#Code
if __name__ == "__main__":
	if len(sys.argv) != 3:
		print(sys.argv)
		print("ERROR :( : syntax is correct ")
		print("Do write what inside brakets")
		print("Syntax example:")
		print(" python name.py 64[block size 64x64] 32[step]")
	else:	
		#params given by the tester
		block = int(sys.argv[1])
		step = int(sys.argv[2])

		img_path = glob.glob("/home/irocs/Desktop/IROCS/NN/Perceptron/images/floor9/*.png")
		gt_path = glob.glob("/home/irocs/Desktop/IROCS/NN/Perceptron/groundtruth/floor9/*.png")

		if img_path == 0:
			sys.exit("ERROR: Image path isn't correct, need to verify on script")
		elif gt_path == 0:
			sys.exit("ERROR: groundtruth path isn't correct, need to verify on script")
		elif block % 16 != 0:
			sys.exit("ERROR: block number isn't correct, verify documentation")
		elif step % 8 != 0:
			sys.exit("ERROR: step number isn't correct, verify documentation")
		else :
			#time execution start
			time_inicial = time.time()
			print("Treino Inicializado ...")
			#convert to lab image used to test

			#Variables Temp
			#Vector to supervise all images
			Vec_gt = []
			#Vector instante supervised
			Vec_inst = []
			#vector to train
			Vec_train = [[],[],[]]
			
			#mean
			X_means = [[],[],[]]
			#stander deviaton
			X_std = [[],[],[]]
			
			#values used to test
			X_test = [[],[],[]]
			#Result of nnp
			X_result = [[],[],[]]

			#label treino
			labels = ["treino_L.txt","treino_a.txt","treino_b.txt"]

			#Calculation of proprity image to train
			for path_im,path_gt in zip(img_path, gt_path):

				#image used to train
				image = cv.imread(path_im)
				rows,cols,dim = image.shape
				#convert to Lab image used to train
				image_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)

				#ground truth of that image
				groundtruth = cv.imread(path_gt,cv.IMREAD_UNCHANGED)
				
				#Classification of classes with groudthruth
				Vec_inst = p_irocs.classification(groundtruth, block,step)
				Vec_gt.extend(Vec_inst)

				#gradient magnitude
				Mag = np.zeros((rows,cols,dim),np.uint16)

				for i in range(dim):
					#init gradient
					prop = p_irocs.Gradient(image_lab[:,:,i])
					#magnitude gradint calculation
					Mag[:,:,i]=prop.Cal_Gradient_Magnitude("sobel")
					#Calcule magnitude gradient propriets mean and std
					X_t, X_means[i], X_std[i] =p_irocs.Cal_properties(Mag[:,:,i],block,step)
					#features vector to train
					Vec_train[i].extend(X_t)

			network = [[],[],[]]
			for i in range(dim):
				#init perceptron network
				network[i] = per.Perceptron(Vec_train[i],Vec_gt)
				network[i].train()
				network[i].saveCharges(labels[i])


			print("--- %s segundos ---" % (time.time() -  time_inicial))	
			