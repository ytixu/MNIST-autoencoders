import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

def load(args):
	(x_train_orig, y_train_orig),(x_test_orig, y_test_orig) = mnist.load_data()
	x_train_orig, x_test_orig = x_train_orig / 255.0, x_test_orig / 255.0
	img_size = x_train_orig.shape[1]

	def _get_zeros(shape):
		if args['random']:
			return np.random.uniform(0, 0.5, shape)
		if args['ones']:
			return np.ones(shape)
		return np.zeros(shape)

	def __const_complete_data(x=[], y=[]):
		if (len(x) > 0):
			x = np.reshape(x, (x.shape[0], -1))
		else:
			x = _get_zeros((y.shape[0], img_size**2))
		if (len(y) > 0):
			y = to_categorical(y)
		else:
			y = _get_zeros((x.shape[0], 10))
		return np.concatenate((x, y), axis=1)

	xy_train = __const_complete_data(x_train_orig, y_train_orig)

	x_train = __const_complete_data(x_train_orig)
	x_xy_train = np.concatenate((x_train, xy_train), axis=0)
	y_train = __const_complete_data([], y_train_orig)

	x_xy_y_train = np.concatenate((x_xy_train, y_train), axis=0)
	# xy_test = __const_complete_data(x_test, y_test_orig)
	x_test = __const_complete_data(x_test_orig)

	y_test_gen = _get_zeros((10,xy_train.shape[1]))
	y_test_gen[:,-10:] = np.eye(10)

	return (x_train_orig, y_train_orig, x_test_orig, y_test_orig), (x_train, y_train, xy_train, x_xy_y_train, x_test, y_test_gen)