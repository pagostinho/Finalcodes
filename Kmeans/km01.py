import projectIrocs as p_irocs
import cv2 as cv
import numpy as np
import math
import sys
import time
import matplotlib.pyplot as plt
import glob

if __name__ == "__main__":
	if len(sys.argv) != 4:
		print(sys.argv)
		print("ERROR :( : syntax is correct ")
		print("Do write what inside brakets")
		print("Syntax example:")
		print(" python name.py 64[block size 64x64] 32[step] sobel[gradtype]")
	else:		
		img_path = glob.glob("/home/irocs/Desktop/IROCS/Kmeans/floor1/*.png")

		if img_path == 0:
			sys.exit("ERROR: Image path isn't correct, need to verify on script")
		elif int(sys.argv[1]) % 16 != 0:
			sys.exit("ERROR: block number isn't correct, verify documentation")
		elif int(sys.argv[2]) % 16 != 0:
			sys.exit("ERROR: step number isn't correct, verify documentation")
		else :

			for path in img_path:
				time_inicial = time.time()
				image = cv.imread(path)
				block = int(sys.argv[1])
				step = int(sys.argv[2])
				grad = sys.argv[3]
				if image is None:
					sys.exit("ERROR: Couldn't read the image")

				#img_splited = image[:dim,:dim]
				lab = cv.cvtColor(image,cv.COLOR_BGR2Lab)
				rows,cols, dim = lab.shape

				#gradient magnitude
				Mag = np.zeros((rows,cols,dim),np.uint16)
				#mean
				X_means = [[],[],[]]
				#stander deviaton
				X_std = [[],[],[]]
				#dirty blocks index
				dirtyIndexs = [[],[],[]]
				#dirty limite
				limit = 7

				#for each channel 
				for i in range(dim):
					prop = p_irocs.Gradient(lab[:,:,i])
					Mag[:,:,i]=prop.Cal_Gradient_Magnitude(grad)
					X_t, X_means[i], X_std[i] =p_irocs.Cal_properties(Mag[:,:,i],block,step)
				
					Z = np.vstack(X_t)
					# convert to np.float32
					Z = np.float32(Z)

					# define criteria and apply kmeans()
					criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
					ret,label,center=cv.kmeans(Z,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

					#Grafique representation
					A = Z[label.ravel()==0]
					B = Z[label.ravel()==1]
					plt.scatter(A[:,0],A[:,1])
					plt.scatter(B[:,0],B[:,1],c = 'r')
					plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
					plt.xlabel('Mean'),plt.ylabel('STD')
					plt.show()

					class1 = np.where((label.ravel()==0)==True)
					class2 = np.where((label.ravel()==1)==True)

					if(len(class1)>len(class2)):
						epiCenter = center[:,0]
						dirtyClass = label.ravel()==1
					else:
						epiCenter = center[:,1]
						dirtyClass = label.ravel()==0

					#normalize bouth axes
					epiCenter[0] = epiCenter[0]/max(X_means[i])*10
					epiCenter[1] = epiCenter[1]/max(X_std[i])*10

					#normalize vector
					Z[:,0] = Z[:,0]/max(Z[:,0])*10 
					Z[:,1] = Z[:,0]/max(Z[:,1])*10

					for j in range(len(Z)):
						if p_irocs.distancia_euclidiana(Z[j],epiCenter)>limit:
							dirtyIndexs[i].append(True)
						dirtyIndexs[i].append(False)

				#channel intersecion
				dirtyFound = (dirtyIndexs[0] and dirtyIndexs[1]) or (dirtyIndexs[0] and dirtyIndexs[2]) or (dirtyIndexs[1] and dirtyIndexs[2])

				print("Dirty Analizing ...")
				dirty = np.zeros((rows,cols,dim),np.uint8)
				

				for l in range(len(Z)):
					if dirtyFound[l] == True : 
						I,J = p_irocs.Indexs(rows,cols,block,l,block/step)
						dirty[I:I+block,J:J+block,:] = 1

				print("--- %s segundos ---" % (time.time() - time_inicial))			
				
				ap_mask = image*dirty

				cv.imshow("Imagem", image)
				cv.imshow("Mascara", ap_mask)

				cv.waitKey(0)
				cv.destroyAllWindows()	
