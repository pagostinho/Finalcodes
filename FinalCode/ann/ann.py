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
	if len(sys.argv) != 5:
		print(sys.argv)
		print("ERROR :( : syntax is correct ")
		print("Do write what inside brakets")
		print("Syntax example:")
		print(" python name.py 64[block size 64x64] 32[step] test.png[test image]")
	else:	
		#params given by the tester
		block = int(sys.argv[1])
		step = int(sys.argv[2])
		
		img_path = glob.glob("/home/irocs/Desktop/IROCS/FinalCode/treino/images/floor9/*.png")
		gt_path = glob.glob("/home/irocs/Desktop/IROCS/FinalCode/treino/groundtruth/floor9/*.png")


		if image is None:
			sys.exit("ERROR: Couldn't read the image")
		elif block % 16 != 0:
			sys.exit("ERROR: block number isn't correct, verify documentation")
		elif step % 8 != 0:
			sys.exit("ERROR: step number isn't correct, verify documentation")
		else :
			#time execution start
			time_inicial = time.time()

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