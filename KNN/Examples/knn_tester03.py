import projectIrocs as p_irocs
import cv2 as cv
import numpy as np
import sys
import random
import glob


#ground truth of that image
ground = cv.imread(sys.argv[1],cv.IMREAD_UNCHANGED)
#image to test
test = cv.imread(sys.argv[2])
#block dimension
block = int(sys.argv[3])
#step for slide block
step = int(sys.argv[4])
#gradient to use
grad = sys.argv[5]
#k nearest neightbors
K = int(sys.argv[6])

#convert to lab image used to test
test_lab = cv.cvtColor(test,cv.COLOR_BGR2Lab)
	
#dimensions of images
rows,cols = ground.shape
dim  = 3
#class classification with ground truth
#classification if is clean(1) or drity(2)
X_o=p_irocs.classification(ground,block,step)
#Values of the classification on np formatq
c_out = 1
#creat some tempor vars
#Values used to train
X_train = [[],[],[]]
#Result of knn
X_result = [[],[],[]]
#values used to test
X_test = [[],[],[]]
#dirty mask operator
dirty = np.zeros((rows,cols,dim),np.uint16)
#hist to calculate accuracy
hits= [[],[],[]]

#Test img
for n in range(dim):
	#class gradient init
	p_t = p_irocs.Gradient(test_lab[:,:,n])
	#magnitude calculation
	p_t.Cal_Gradient_Magnitude(grad) 
	#proprities calculation, mean and std
	X_test[n]=p_irocs.Cal_properties(p_t.Mag,block,step)

#read path to all images to test random
#NOTE: 	this image are pratic the same just with diferente luma, because of that
#		is used the same ground truth
img_path = glob.glob("/home/irocs/Desktop/IROCS/Test/train/*.png")

for path in img_path:
	X_out = np.array(X_o*c_out,np.int32)

	#image used to train
	image = cv.imread(path)
	#convert to Lab image used to train
	image_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
	dirty = np.zeros((rows,cols,dim),np.uint16)
	#for each channel 
	for i in range(dim):
		#train
		#class gradient init
		prop = p_irocs.Gradient(image_lab[:,:,i])
		#magnitude calculation
		prop.Cal_Gradient_Magnitude(grad) 
		#proprities calculation, mean and std
		X_i = p_irocs.Cal_properties(prop.Mag,block,step)
		X_train[i].extend(X_i)
		#features to train
		X_in = np.array(X_train[i],np.int32)
		#train result KNN
		for j in range(len(X_test[0])):
			X_result[i].append(p_irocs.KNN(X_in,X_out,X_test[i][j],K))
	#found the clean and dirty spots
	
	for l in range(len(X_result[0])):
		#mean of classes
		classe = float(X_result[0][l] + X_result[1][l] + X_result[2][l])/3
		#over 1.5 is class 2, that means is dirty spot
		if classe > 1.5:
			#found de initial indices
			pos = l
			if c_out > 1 : 
				pos = l/c_out

			print(pos)
			I,J = p_irocs.Indexs(rows,cols,block,pos,block/step)
			print(I,J)
			dirty[I:I+block,J:J+block,:] = 255

	cv.imshow("Dirty spots",dirty)
	X_r = np.array(X_result,np.int32)
	print(X_r)
	#count de correct predictions made
	for i in range(len(X_result)):
		count = 0
		hits[i] = (X_r[i]==X_out)
		print(hits)
		for j in range(len(X_result[0])):
			if(hits[i][j] == True):
				count+=1
		print("Accuracy channel %d = %f" %(i,(count/float(len(X_result[0])))*100))

	#mask aplication
	ap_mask = test*dirty
	cv.imshow("Aplication Mask",ap_mask)
	cv.waitKey(0)
	cv.destroyAllWindows()	
	#For each image append another ground truth classe clssify
	c_out +=1
	#TODO: Debug is not working concatenate, and mask identification
cv.waitKey(0)
cv.destroyAllWindows()	