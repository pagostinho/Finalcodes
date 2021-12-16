#Data
import projectIrocs as p_irocs
import percepton as per
import sys
import time
import cv2 as cv
import numpy as np

#Zona de Codigo
if __name__ == "__main__":
	if len(sys.argv) != 4:
		print(sys.argv)
		print("ERROR :( : syntax is correct ")
		print("Do write what inside brakets")
		print("Syntax example:")
		print(" python name.py 64[block size 64x64] 32[step] test.png[test image]")
	else:	
		#block dimension
		block = int(sys.argv[1])
		#step for slide block
		step = int(sys.argv[2])

		#image to test
		test = cv.imread(sys.argv[3])
		if test is None:
			sys.exit("ERROR: Couldn't read the image")
		elif block % 16 != 0:
			sys.exit("ERROR: block number isn't correct, verify documentation")
		elif step % 8 != 0:
			sys.exit("ERROR: step number isn't correct, verify documentation")
		else :
			#time execution start
			time_inicial = time.time()
			print("Teste Inicializado ...")
			#convert to lab image used to test
			test_lab = cv.cvtColor(test,cv.COLOR_BGR2Lab)
			#dimensions of images
			rows,cols,dim = test.shape

			#values used to test
			X_test = [[],[],[]]
			#Result of nnp
			X_result = [[],[],[]]

			#label treino
			labels = ["treino_L.txt","treino_a.txt","treino_b.txt"]
			network = [[],[],[]]

			#Properties of test image
			for i in range(dim):

				#class gradient init
				p_t = p_irocs.Gradient(test_lab[:,:,i])
				#magnitude calculation
				p_t.Cal_Gradient_Magnitude("sobel") 
				#proprities calculation, mean and std
				X_test[i]=p_irocs.Cal_properties(p_t.Mag,block,step)

			#train and test for each image channel
			for i in range(dim):
				network[i] = per.Perceptron(X_test[i],0)
				network[i].loadCharges(per.LoadCharges(labels[i]),3)
				for j in range(len(X_test[0][0])):
					X_result[i].append(network[i].test(X_test[i][0][j]))

			dirty = np.zeros((rows,cols,dim),np.uint8)
			for l in range(len(X_result[0])):
				#mean of classes
				classe = X_result[0][l] + X_result[1][l] + X_result[2][l]
				if classe > 1.5:
					I,J = p_irocs.Indexs(rows,cols,block,l,block/step)
					dirty[I:I+block,J:J+block,:] = 1

			print("--- %s segundos ---" % (time.time() - time_inicial))

			ap_mask = test*dirty
			cv.imwrite('mask.png', dirty*255)

