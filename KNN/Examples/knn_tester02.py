import projectIrocs as p_irocs
import cv2 as cv
import numpy as np
import sys
import random
import glob

#ground truth of that image
ground = cv.imread(sys.argv[2],cv.IMREAD_UNCHANGED)
#image to test
test = cv.imread(sys.argv[3])
#block dimension
block = int(sys.argv[4])
#step for slide block
step = int(sys.argv[5])
#gradient to use
grad = sys.argv[6]
#k nearest neightbors
K = int(sys.argv[7])

#creat some tempor vars
#Result of knn
X_result = [[],[],[]]
#corret values
X_value = [[]]

#class classification with ground truth
#classification if is clean(1) or drity(2)
X_o=p_irocs.classification(ground,block,step)
X_out = np.array(X_o,np.int32)


#print("Samples class clean = %d" %X_o.count(1))
N_c = X_o.count(1)
#print("Samples class dirty = %d" %X_o.count(2))
N_d = X_o.count(2)

#print(np.where(X_out==1))
#Index of classe clean
I_c = np.where(X_out==1)
#print(np.where(X_out==2))
#Index of class dirty
I_d = np.where(X_out==2)


img_path = glob.glob("/home/irocs/Desktop/IROCS/Test/train/*.png")

for path in img_path:
	#image used to train
	image = cv.imread(path)
	#convert to Lab image used to train
	image_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
	#convert to lab image used to test
	test_lab = cv.cvtColor(test,cv.COLOR_BGR2Lab)
	#dimensions of images
	rows,cols, dim = image_lab.shape
	#for each channel 
	for i in range(dim):
		#train
		#class gradient init
		prop = p_irocs.Gradient(image_lab[:,:,i])
		#magnitude calculation
		prop.Cal_Gradient_Magnitude(grad) 
		#proprities calculation, mean and std
		X_i=p_irocs.Cal_properties(prop.Mag,block,step)
		#features to train
		X_in = np.array(X_i,np.int32)
		#test
		#class gradient init
		p_t = p_irocs.Gradient(test_lab[:,:,i])
		#magnitude calculation
		p_t.Cal_Gradient_Magnitude(grad) 
		#proprities calculation, mean and std
		X=p_irocs.Cal_properties(p_t.Mag,block,step)
		#features to train
		X_test = np.array(X,np.int16)
		#print(X_test)
		for j in range(len(X_test)):
			X_result[i].append(p_irocs.KNN(X_in,X_out,X_test[j],K))

	#dirty mask operator
	dirty = np.zeros((rows,cols,dim),np.uint8)

	#found the clean and dirty spots
	for l in range(len(X_result[0])):
		#mean of classes
		classe = (X_result[0][l] + X_result[1][l] + X_result[2][l])/3
		#over 1.5 is class 2, that means is dirty spot
		if classe > 1.5:
			#found de initial indices
			I,J = p_irocs.Indexs(rows,cols,block,l,block/step)
			dirty[I:I+block,J:J+block,:] = 1

	#hist to calculate accuracy
	hits= [[],[],[]]
	X_r = np.array(X_result,np.int32)
	#count de correct predictions made
	for i in range(len(X_result)):
		count = 0
		hits[i] = (X_r[i]==X_out[i])
		for j in range(len(X_result[0])):
			if(hits[i][j] == True):
				count+=1

		print("Accuracy channel %d = %f" %(i,(count/float(len(X_result[0])))*100))



	ap_mask = test*dirty
	cv.imshow("Aplication Mask",ap_mask)

#TODO: DEBUG and TEST

cv.waitKey(0)
cv.destroyAllWindows()	