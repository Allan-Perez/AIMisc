import numpy as np

#A computational graph is a way to represent computaitons, following the ancient 
#concept divide et impera. Basically, we break up a big computation like 
#f(x) = e**(x+2) * 1/x**2 to smaller computations that are connected somehow so that 
# the result (output) of one is the input of the next (hence a directed graph):
# u = x+2 | v = x**2 | w = 1/v | k = e**u | f(x)=kw. 
# First we compute u and v, then w and k, finally f(x).
# It's useful to think of computational graph as a piping system we build, and 
# computing an actual value is just making water flow through the pipes. 

# In this example I'll try to make a simple linear transformation plus an origin shift
# y=Ax+b, and applying a function to that operation f(y). 
# w = Ax 
# y = w+b
#Later, I'll compute the gradient
# of the said function.



class Graph():
	def __init__(self):
		self.ops = []
		self.placeholders=[]
		self.params=[]

class Op:
	def __init__(self, graph, op=None, in_nodes=[]):
		if op==None:
			raise Exception("An op node must have an operation")
		self.op = op
		self.in_nodes = in_nodes
		for in_node in in_nodes:
			in_node.out_nodes.append(self) 

		self.out_nodes = []

		graph.ops.append(self)

	def compute(self, *args):
		return self.op(*args)

class Placeholder:
	def __init__(self, graph):
		self.out_nodes = []
		graph.placeholders.append(self)

class Param:
	def __init__(self, graph, val):
		self.out_nodes = []
		self.val = val
		graph.params.append(self)

def flow(outop, feed_dict={}):
	graph = graph_reconstruct(outop)
	print('Evaluating the graph...')
	for node in graph:
		if isinstance(node, Placeholder):
			node.outval = feed_dict[node]
		elif isinstance(node, Param):
			node.outval = node.val
		elif isinstance(node, Op):
			node.invals = [in_node.outval for in_node in node.in_nodes] 
			node.outval = node.compute(*node.invals)
		#print('Flowing node:', node, '\tVal:', node.outval)
	return outop.outval

def graph_reconstruct(node):
	graph = []

	def reconstruct(node_):
		if isinstance(node_, Op):
			for in_node in node_.in_nodes:
				reconstruct(in_node)
		graph.append(node_)
	print('reconstructing the graph...')
	reconstruct(node)
	print('reconstructed!')
	return graph

def main():
	x_vect = np.array([2,1])

	matmult = lambda A,x: A.dot(x)
	add = lambda w,b: w+b

	graph1 = Graph()
	A = Param(graph1, np.array([[1,2],[3,4]]))
	b = Param(graph1, np.array([1,1]))
	x = Placeholder(graph1)

	w = Op(graph1, matmult, [A,x])
	y = Op(graph1, add, [w,b])

	output_vect = flow(y, feed_dict={x: x_vect})
	print('Input vector:',  x_vect)
	print('Output vector:',  output_vect)

if __name__ == '__main__':
	main()