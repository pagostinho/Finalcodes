import numpy as np
import percepton as per


X_in = [[200,120], [100,110], [120,110], [90,100],[180,130],[193,124],
		[130,130],[111,101],[140,100],[130,122],[199,140],[180,120],[191,142],
		[100,100],[111,101],[122,111],[123,134],[140,111],[133,122]]
X_out = [1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0]

network = per.Perceptron(X_in,X_out)
print(network.train())
print(network.test([193,120]))
print(network.charges)
network.saveCharges("charges.txt")

print(per.LoadCharges("charges.txt"))
