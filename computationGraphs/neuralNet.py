#this is an example of a feedforward fully-connected multilayer peprceptron
import numpy as np
import computationGraphUtils as cgu
#Graph, Op, Placeholder, Param, flow, graph

def create_layer(graph, n_neurons, in_dims, previous_lay, af, bias):
	if af == None: 
		af = lambda x: np.max(x,0)

	matmult = lambda A,x: A.dot(x)
	add = lambda w,b: np.add(w,b)

	A = cgu.Param(graph, np.random.rand(n_neurons, in_dims))
	b = cgu.Param(graph, np.ones((n_neurons,1)))
	
	if bias:
		wa_ = cgu.Op(graph, matmult, [A,previous_lay])
		z = cgu.Op(graph, add, [wa_,b])
		a = cgu.Op(graph, af, [z])
	else:
		wa_ = cgu.Op(graph, matmult, [A,previous_lay])
		z = cgu.Op(graph, add, [wa_,np.zeros((n_neurons,1))])
		a = cgu.Op(graph, af, [z])
	return a

class Mlp:
	def __init__(self):
		self.graph = cgu.Graph()
		self.layers = []
		self.dims = []		

	def add_layer(self, n_neurons, actf=None, in_dims=None, bias=True):
		if in_dims != None:
			if not isinstance(in_dims,int):
				raise Exception('in_dims must be an integer, instead it is ',type(in_dims))
			self.in_x_label = cgu.Placeholder(self.graph)
			self.layers.append(create_layer(self.graph, n_neurons, in_dims, self.in_x_label, actf, bias))
			self.dims.append(n_neurons)

		else: 
			if len(self.layers) == 0:
				raise Exception('To add a hidden layer, firstly you need to add an input layer')
			self.layers.append(create_layer(self.graph, n_neurons, self.dims[-1], self.layers[-1], actf, bias))
			self.dims.append(n_neurons)

	def compile(self, loss_func, optimizer=None):
		loss_func = self.lossfunction(loss_func)
		# TODO: optimization computation (gradient of computaiton graph)
		self.label = cgu.Placeholder(self.graph)
		self.loss = cgu.Op(self.graph, loss_func, [self.label, self.layers[-1]])

	def predict(self, x):
		prediction = cgu.flow(self.layers[-1], feed_dict={self.in_x_label:x})
		return prediction

	def fit(self, x_train, y_train):
		# For this method to run, we first need to fix the method `compile`
		pass

	def lossfunction(self, func_name):
		if func_name == 'mean_squared_error':
			return lambda y,y_hat: ((y-y_hat)**2).sum()
		raise Exception('The loss function '+func_name+' is not known.')
		

def activation(func_name, alpha=None):
	def check_alpha():
		if alpha == None:
			raise Exception('A value for alpha is needed: '+func_name)

	if func_name == 'relu':
		return lambda x: np.maximum(x,0)
	elif func_name == 'elu':
		check_alpha()
		return lambda x: np.maximum(x, alpha*(np.exp(x)-1))
	elif func_name == 'leaky_relu':
		check_alpha()
		return lambda x: np.maximum(x, alpha*x)
	elif func_name == 'sigmoid':
		return lambda x: (1-np.exp(-x))**-1
	elif func_name == 'tanh':
		return lambda x: (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
	elif func_name == 'softmax':
		return lambda x: np.exp(x - x.max(axis=None, keepdims=True))/np.exp(x - x.max(axis=None, keepdims=True)).sum(axis=None, keepdims=True)
	raise Exception('The activation function '+func_name+' is not known.')

def main():
	in_testing = np.random.rand(784,1)
	model = Mlp()
	model.add_layer(16, activation('relu'), in_dims=784)
	model.add_layer(32, activation('relu'))
	model.add_layer(64, activation('relu'))
	model.add_layer(32, activation('relu'))
	model.add_layer(16, activation('relu'))
	model.add_layer(5, activation('softmax'))
	model.compile('mean_squared_error')
	print('Prediction: ',model.predict(in_testing))

if __name__ == '__main__':
	main()