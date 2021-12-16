import random

#Perception Implementation

class Perceptron:

	def __init__(self, samples, outputs, tax = 0.1, epochs = 1000, threshold = -1):
		self.samples = samples
		self.outputs = outputs
		self.tax = tax
		self.epochs = epochs
		self.threshold = threshold
		self.N = len(samples)
		self.atributes = len(samples[0])
		self.charges = []

	def train(self):

		for samples in self.samples:
			#put in the first colum -1
			samples.insert(0,-1)

		for i in range(self.atributes):
			self.charges.append(random.random())

		self.charges.insert(0,self.threshold)
		#epochs counter
		N_epochs = 0

		#Initiral error nonexistent
		error = False

		while True:
			for i in range(self.N):
				u=0
				for j in range(self.atributes+1):
					u += self.charges[j]*self.samples[i][j]
				#Obtain network output
				y = self.step(u)
				if y != self.outputs[i]:
					#Error calcualtion
					e_aux = self.outputs[i]-y
					#Ajustement of the charges for each sample element
					for j in range(self.atributes+1):
						self.charges[j] = self.charges[j] + self.tax*e_aux*self.samples[i][j]
					error = True

			N_epochs +=1

			if not error or N_epochs>self.epochs:
				break

		return N_epochs -1

	def test(self,sample):
		sample.insert(0,-1)
		u  = 0
		for i in range(self.atributes + 1):
			u += self.charges[i] * sample[i]
		y = self.step(u)
		return y


	def saveCharges(self,name):
		f = open(name,'w')
		#IMPORTANTE: Frist number of the file is the number o lines to read,
		#			that means the number of charges of your neural network
		f.write(str(len(self.charges))+"\n")
		for i in range(len(self.charges)):
			f.write(str(self.charges[i])+"\n")
		f.close()

	def step(self,u):
		if u >= 0 :
			return 1
		return 0 

#This function have the purpose of read form  a file the number of charges saved on
#on a random file
def LoadCharges(name):
	f = open(name,'r')
	#read the number of charges to read
	N = int(f.readline())
	charges = []
	for i in range(N):
		charges.append(float(f.readline()))
	return charges

