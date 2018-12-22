import numpy as np

def SGD(learning_rate=1e-2, decay=1e-6, momentum=0.9):
	"""TODO: everything basically. Firstly, get the gradient of the 
	cost function (get numerically by changing d=1e-3 every param of J in terms
	of weights and biases). Secondly, update weight matrices and bias vectors 
	with the update formula: wi := wi - lr*dJ/dwi, where the value of 
	dJ/dwi is in the gradient of J.  
	Parallely, the algorithm of chunking the dataset(i/o) into batches.
	Lastly, it has to be put into a while loop, until the cost function 
	reaches a minimum threshold or a max numbers of epochs. """
	pass 
