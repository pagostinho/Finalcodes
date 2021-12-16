import projectIrocs as p_irocs
import cv2 as cv
import numpy as np
import sys

image = cv.imread(sys.argv[1])
ground = cv.imread(sys.argv[2])
test = cv.imread(sys.argv[3])
block = int(sys.argv[4])
step = int(sys.argv[5])
grad = sys.argv[6]
K = int(sys.argv[7])
cv.imshow("Image", image)
cv.imshow("Grouwnd Truth",ground)

#img_splited = image[:dim,:dim]
image_lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
test_lab = cv.cvtColor(test,cv.COLOR_BGR2Lab)
rows,cols, dim = image_lab.shape

#Convolution
#kernel = np.ones((5,5),np.float32)/25
#lab = cv.filter2D(image_lab,-1,kernel)

#percent of block train and test

#C_t=[]
#p=[[],[],[]]
X_result = [[],[],[]]
X_value = [[],[],[]]
#for each channel 
for i in range(dim):
	#train
	#class gradient init
	prop = p_irocs.Gradient(image_lab[:,:,i])
	#magnitude calculation
	prop.Cal_Gradient_Magnitude(grad) 
	#proprities calculation, mean and std
	X_i=p_irocs.Cal_properties(prop.Mag,block,step)
	#classification if is clean(1) or drity(2)
	X_o=p_irocs.classification(ground[:,:,i],block,step)
	#features to train
	X_in = np.array(X_i,np.int32)
	#print(X_in)
	X_out = np.array(X_o,np.int32)
	#print(X_out)
	X_value[i].append(X_o)
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
dirty = np.zeros((rows,cols,dim),np.uint8)


for l in range(len(X_result[0])):
	classe = (X_result[0][l] + X_result[1][l] + X_result[2][l])/3
	
	if classe > 1.5:
		I,J = p_irocs.Indexs(rows,cols,block,l,block/step)
		dirty[I:I+block,J:J+block,:] = 1

X_r = np.array(X_result,np.int32)
X_v = np.array(X_value,np.int32) 

hits= [[],[],[]]

for i in range(len(X_result)):
	count = 0
	hits[i] = (X_r[i]==X_v[i])
	for j in range(len(X_result[0])):
		if(hits[i][0][j] == True):
			count+=1

	print("Accuracy channel %d = %f" %(i,(count/float(len(X_result[0])))*100))



ap_mask = test*dirty
cv.imshow("Aplication Mask",ap_mask)

#TODO: DEBUG and TEST

cv.waitKey(0)
cv.destroyAllWindows()	