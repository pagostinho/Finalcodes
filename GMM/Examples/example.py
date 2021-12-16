import numpy as np

from sklearn.mixture import GaussianMixture

X_train=np.array([[100,90],[100,95],[90,90],[95,90],[10,10],[20,25],[30,25]],np.int8)
gmm = GaussianMixture(n_components=2)

gmm.fit(X_train)
X_test = np.array([[100,100],[100,100],[10,10],[100,100],[105,95],[102,98],[20,20]],np.int8)


print(gmm.predict(X_test))