#Zona de dados
import projectIrocs as p_irocs
import percepton as per
import cv2 as cv
import numpy as np
import sys
import glob
import time
import percepton as per

#Zona de Codigo
if __name__ == "__main__":
	if len(sys.argv) != 5:
		print(sys.argv)
		print("ERROR :( : syntax is correct ")
		print("Do write what inside brakets")
		print("Syntax example:")
		print(" python name.py 64[block size 64x64] 32[step] sobel[gradtype] test.png[test image]")
	else:	
		img_path = glob.glob("/home/irocs/Desktop/IROCS/NN/Perceptron/images/floor9/*.png")
		gt_path = glob.glob("/home/irocs/Desktop/IROCS/NN/Perceptron/groundtruth/floor9/*.png")

		imgteste_path = glob.glob("/home/irocs/Desktop/IROCS/NN/Perceptron/teste/images/floor9/*.png")
		gtteste_path = glob.glob("/home/irocs/Desktop/IROCS/NN/Perceptron/teste/groundtruth/floor9/*.png")
		
		if img_path == 0:
			sys.exit("ERROR: Image path isn't correct, need to verify on script")
		elif gt_path == 0:
			sys.exit("ERROR: groundtruth path isn't correct, need to verify on script")
		elif int(sys.argv[1]) % 16 != 0:
			sys.exit("ERROR: block number isn't correct, verify documentation")
		elif int(sys.argv[2]) % 8 != 0:
			sys.exit("ERROR: step number isn't correct, verify documentation")
		else :

			#block dimension
			block = int(sys.argv[1])
			#step for slide block
			step = int(sys.argv[2])
			#gradient to use
			grad = sys.argv[3]

			#image to test
			test = cv.imread(sys.argv[4])
			if test is None:
				sys.exit("ERROR: Couldn't read the image")
			#convert to lab image used to test
			test_lab = cv.cvtColor(test,cv.COLOR_BGR2Lab)
			#dimensions of images
			rows,cols,dim = test.shape

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
			#gradient magnitude
			Mag = np.zeros((rows,cols,dim),np.uint16)
			#values used to test
			X_test = [[],[],[]]
			#Result of nnp
			X_result = [[],[],[]]
			accuracy = 0
			time_final = 0
			#Calculation of proprity image to train
			for path_im,path_gt in zip(img_path, gt_path):
				#time execution start
				time_inicial = time.time()

				#Variables Temp
				#vector final results
				Vec_rf = []
				#Result of knn
				X_result = [[],[],[]]

				#ground truth of that image
				groundtruth = cv.imread(path_gt,cv.IMREAD_UNCHANGED)

				#Classification of classes with groudthruth
				Vec_inst = p_irocs.classification(groundtruth, block,step)
				Vec_gt.extend(Vec_inst)

				#image used to train
				image = cv.imread(path_im)
				#convert to Lab image used to train
				image_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)

				for i in range(dim):
					#init gradient
					prop = p_irocs.Gradient(image_lab[:,:,i])
					#magnitude gradint calculation
					Mag[:,:,i]=prop.Cal_Gradient_Magnitude(grad)
					#Calcule magnitude gradient propriets mean and std
					X_t, X_means[i], X_std[i] =p_irocs.Cal_properties(Mag[:,:,i],block,step)
					#features vector to train
					Vec_train[i].extend(X_t)

			network = [[],[],[]]
			for i in range(dim):
					#init perceptron network
					network[i] = per.Perceptron(Vec_train[i],Vec_gt)
					network[i].train()
			

			#Counter to calculate de accuracy
			count = 0	
			fail = 0	
			for pathtest_im,pathtest_gt in zip(imgteste_path, gtteste_path):
				count = count+1
				image = cv.imread(pathtest_im)
				test_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
				X_result = [[],[],[]]
				#Properties of test image
				for i in range(dim):

					#class gradient init
					p_t = p_irocs.Gradient(test_lab[:,:,i])
					#magnitude calculation
					p_t.Cal_Gradient_Magnitude(grad) 
					#proprities calculation, mean and std
					X_test[i]=p_irocs.Cal_properties(p_t.Mag,block,step)

				#train and test for each image channel
				for i in range(dim):
					for j in range(len(X_test[0][0])):
						X_result[i].append(network[i].test(X_test[i][0][j]))


				#print("--- %s segundos ---" % (time_final))	

				vec_rf = np.zeros((1,len(X_result[0])),np.uint8)
				
				#dirty mask
				dirty = np.zeros((rows,cols,dim),np.uint8)
				for l in range(len(X_result[0])):
					#mean of classes
					classe = X_result[0][l] + X_result[1][l] + X_result[2][l]
					if classe >=1.5:
						vec_rf[0,l] = 1
						I,J = p_irocs.Indexs(rows,cols,block,l,block/step)
						dirty[I:I+block,J:J+block,:] = 1


				#ground truth of that image
				groundtruth = cv.imread(pathtest_gt,cv.IMREAD_UNCHANGED)

				#Classification of classes with groudthruth
				Vec_inst = p_irocs.classification(groundtruth, block,step)
				fail = np.count_nonzero((vec_rf - vec_rf * Vec_inst)==1)
				accuracy= accuracy + float(np.count_nonzero((vec_rf * Vec_inst) == 1))/float(Vec_inst.count(1))

				ap_mask = image*dirty

				cv.imshow("Imagem", image)
				cv.imshow("Mascara", ap_mask)

				cv.waitKey(0)
				cv.destroyAllWindows()
				print(count)

			print("Accuracy = " + str(accuracy/count))
			print("Fails = " + str(fail/count))

