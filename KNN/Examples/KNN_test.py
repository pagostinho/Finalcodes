import projectIrocs as p_irocs
import numpy as np

X_in = [[200,120], [100,110], [120,110], [90,100],[180,130],[193,124],
		[130,130],[111,101],[140,100],[130,122],[199,140],[180,120],[191,142],
		[100,100],[111,101],[122,111],[123,134],[140,111],[133,122]]
X_out = [2,1,1,1,2,2,1,1,1,1,2,2,2,1,1,1,1,1,1]

X_test = [[[220,180],[110,110],[230,170],[140,122]],
			[[210,185],[100,99],[134,130],[90,80]],
			[[222,180],[210,190],[100,100],[120,110]]]

X_result = [[],[],[]]

X_o = np.array(X_out,np.int32)
X_i = np.array(X_in, np.int32)

c2 = X_out.count(2) 

i2 = np.where(X_o==2)
i1 = np.where(X_o==1)

i1r = np.random.choice(i1[0].shape[0],c2,replace=False)


X1 = []
X2 = []

for i in range(c2):
	X1.append(X_in[i1[0][i1r[i]]])
	X1.append(X_in[i2[0][i]])
	X2.append(1)
	X2.append(2)

for j in range(3):
	for i in range(len(X_test[0])):
		X_result[j].append(p_irocs.KNN(X1,X2,X_test[j][i],3))

print(X_result)