import os 
import gzip as gz
import numpy as np
import optimizers as optim
import matplotlib.pyplot as plt
from neuralNet import Mlp, activation


def get_data(file_dir, im_dims, sample_size, ini_bytes, dtype=np.float32):
	raw_data = gz.open(file_dir, 'r')

	read_limit = im_dims[0]*im_dims[1]*im_dims[2]*sample_size if sample_size>0 else sample_size
	raw_data.read(ini_bytes)
	buffer_ = raw_data.read(read_limit)
	dataframe = np.frombuffer(buffer_, dtype=np.uint8).astype(dtype)

	return dataframe.reshape(sample_size, im_dims[0]*im_dims[1], im_dims[2])
	

def one_hot_transform(data):
	# data must be 1d arrays with int values 
	# returns 2d array (matrix) with each row being the value of int 
	# and each column the value of the onehot encoding
	assert type(data) == np.ndarray
	range_ = data.max() - data.min()
	def onehot_encoding(i):
		row = np.zeros(range_)
		row[i-1] += 1
		return row
	return np.array(list(map(onehot_encoding, data)))


def main_data(sample_size=-1):
	train_images = '../data/MNIST/train-images-idx3-ubyte.gz'
	train_labels = '../data/MNIST/train-labels-idx1-ubyte.gz'
	test_images = '../data/MNIST/t10k-images-idx3-ubyte.gz'
	test_labels = '../data/MNIST/t10k-labels-idx1-ubyte.gz'


	im_dims = (28,28,1)
	la_dims = (1,1,1)
	ini_im_bytes = 16
	ini_la_bytes = 8


	df_train_im = get_data(train_images, im_dims, sample_size, ini_im_bytes)
	df_train_la = get_data(train_labels, la_dims, sample_size, ini_la_bytes, np.int64)

	df_test_im = get_data(test_images, im_dims, sample_size, ini_im_bytes)
	df_test_la = get_data(test_labels, la_dims, sample_size, ini_la_bytes, np.int64)

	df_train_la = one_hot_transform(df_train_la.squeeze())
	df_test_la = one_hot_transform(df_test_la.squeeze())
	return (df_train_im, df_train_la), (df_test_im, df_test_la)

def create_mlp():
	model = Mlp()
	model.add_layer(16, activation('relu'), in_dims=784)
	model.add_layer(32, activation('relu'))
	model.add_layer(64, activation('relu'))
	model.add_layer(32, activation('relu'))
	model.add_layer(16, activation('relu'))
	model.add_layer(10, activation('softmax'))
	optimizer = optim.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9)
	model.compile(loss='mean_squared_error', optimizer='sgd')
	return model


def main():
	df_train, df_test = main_data()
	model = create_mlp()
	print(model.predict(df_train[0][0]), df_train[1][0])
	
	#print(df_train[0].shape)
	#print(df_train[1].shape)


if __name__ == '__main__':
	main()