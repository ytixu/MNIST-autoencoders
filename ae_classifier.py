import numpy as np
import matplotlib.pyplot as plt

import utils.parser as parser
# parse args
args = parser.parse()

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, Reshape
from keras import backend as K

import utils.data_loader as data_loader
from autoencoders.models import *
import utils.viz as viz

# load data
data1, data2 = data_loader.load(args)
x_train_orig, y_train_orig, x_test_orig, y_test_orig = data1
x_train, y_train, xy_train, x_xy_y_train, x_test, y_test_gen = data2

IMG_SIZE = x_train_orig.shape[1]
INPUT_SIZE = x_train.shape[1]
LATENT_SIZE = 32
EPOCHS = 50
EPOCHS_COMPLETE = 5
BATCH_SIZE = 64

##############
# Autoencoder
if args['model'] == 'flatten':
	ae_model = Flatten_AE(INPUT_SIZE, LATENT_SIZE)
elif args['model'] == 'dense_cnn':
	ae_model = Dense_CNN_AE(INPUT_SIZE, LATENT_SIZE, IMG_SIZE)
	EPOCHS = 12
	BATCH_SIZE = 128
elif args['model'] == 'cnn':
	ae_model = CNN_AE(INPUT_SIZE, LATENT_SIZE, IMG_SIZE)
	EPOCHS = 20
	BATCH_SIZE = 128

ae_model.ae.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['mae'])

if args['load_path']:
	ae_model.ae.load_weights(args['load_path'])
else:
	ae_model.ae.fit(x_xy_y_train, x_xy_y_train, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE)
	ae_model.ae.save(args['save_path'])

score = ae_model.ae.evaluate(x_test, x_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]

def relu_advanced(x):
    return K.relu(x, max_value=1)


##############
def feature_extraction():
	ext_train = ae_model.encoder.predict(x_train)
	ext_test = ae_model.encoder.predict(x_test)

	ext_input = Input(shape=(LATENT_SIZE,), name='ext_input')
	ext_output = Dense(10, activation='softmax')(ext_input)

	ext = Model(ext_input, ext_output)
	ext.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	ext.fit(ext_train, y_train_orig, epochs=EPOCHS_COMPLETE, shuffle=True, batch_size=16)

	score = ext.evaluate(ext_test, y_test_orig, verbose=0)
	print 'Test loss:', score[0]
	print 'Test accuracy:', score[1]



##############
print 'Completion function'
def get_complete_func(x_train_, y_train_):

	a_train = ae_model.encoder.predict(x_train_)
	b_train = ae_model.encoder.predict(y_train_)

	g_input = Input(shape=(LATENT_SIZE,), name='g_input')
	g_dense = Dense(LATENT_SIZE, activation='sigmoid', use_bias=False, name='g_dense')
	g_layer = g_dense(g_input)

	g = Model(g_input, g_layer)

	g.compile(optimizer='adam',
					loss='mean_squared_error',
					metrics=['mae'])

	g.fit(a_train, b_train, epochs=EPOCHS_COMPLETE, shuffle=True, batch_size=16)

	g_layer_ = g_dense(ae_model.encoder_layer)
	decoder_layer_ = ae_model.decoder_layers(g_layer_)
	
	return decoder_layer_, g_layer_

def vector_addition(x_train_, y_train_):
	a_train = ae_model.encoder.predict(x_train_)
	b_train = ae_model.encoder.predict(y_train_)
	diff = b_train - a_train
	return np.mean(diff, axis=0), np.std(diff, axis=0)


def get_failed_classes(F):
	predict_labels = F.predict(x_test)
	predict_labels = predict_labels.argmax(axis=-1)
	inc_idx = np.nonzero(predict_labels != y_test_orig)[0]
	print '# of incorrect predictions: %d/%d' %(len(inc_idx), len(y_test_orig))
	
	inc = np.zeros((100,IMG_SIZE,IMG_SIZE))
	for i in inc_idx:
		inc[predict_labels[i]*10+y_test_orig[i]] = np.reshape(x_test[i,:-10], (IMG_SIZE,IMG_SIZE))

	viz.plot_matrix(inc, 'Examples of failed predictions (x=True, y=Predicted)')

##############
def classification(x_train_, xy_train_):
	class_layer, _ = get_complete_func(x_train_, xy_train_)

	drop_x = Lambda(lambda x: x[:,-10:], output_shape=(10,), name='drop_lambda')
	class_output = Activation('softmax', name='softmax')

	F_layers = class_output(drop_x(class_layer))
	F = Model(ae_model.encoder_input, F_layers)
	F.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	get_failed_classes(F)

	score = F.evaluate(x_test, y_test_orig, verbose=0)
	print 'Forward network---'
	print 'Test loss:', score[0]
	print 'Test accuracy:', score[1]

	x_input = ae_model.encoder.predict(x_test)
	x_pred = x_input[:] + vector_addition(x_train_, xy_train_)[0]
	F_input = Input(shape=(LATENT_SIZE,), name='va_input')
	F_layers = class_output(drop_x(ae_model.decoder_layers(F_input)))
	F = Model(F_input, F_layers)

	F.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])
	

	score = F.evaluate(x_pred, y_test_orig, verbose=0)
	print 'Vector addition---'
	print 'Test loss:', score[0]
	print 'Test accuracy:', score[1]



##############
print 'Generation'

def neighbours(z, diff_std, layers=1):
	z_pred = np.zeros(((layers*2+1)**2, z.shape[-1]))
	z_pred[0] = z
	for layer in range(1,layers+1):
		n = layer*2
		N = (n+1)**2-1
		for i in range(n*4):
			z_pred[N-i] = np.random.normal(loc=z, scale=diff_std*layer*0.5, size=None)
	return z_pred

def transition(F, z, mix=0.2):
	new_z = np.zeros((100,LATENT_SIZE))
	for i in range(10):
		new_z[i*10:(i+1)*10,:] = z[i]
		new_z[i*10:(i+1)*10] = (1.0-mix)*new_z[i*10:(i+1)*10] + mix*z

	gen = F.predict(new_z)
	viz.plot_matrix(gen)



def generation(y_train_, xy_train_):
	gen_layer, z_layer = get_complete_func(y_train_, xy_train_)
	drop_y = Lambda(lambda x: x[:,:-10], output_shape=(IMG_SIZE**2,), name='drop_lambda')
	gen_output = Reshape((IMG_SIZE, IMG_SIZE))

	gen_layers = gen_output(drop_y(gen_layer))
	F = Model(ae_model.encoder_input, gen_layers)
	F.compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy'])

	gen = F.predict(y_test_gen)
	viz.plot(gen)

	# Vector addition
	diff_mean, diff_std = vector_addition(y_train_, xy_train_)

	x_input = ae_model.encoder.predict(y_test_gen)
	x_pred = x_input[:] + diff_mean
	F_input = Input(shape=(LATENT_SIZE,), name='va_input')
	F_layers = gen_output(drop_y(ae_model.decoder_layers(F_input)))
	F = Model(F_input, F_layers)

	F.compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy'])

	gen = F.predict(x_pred)
	viz.plot(gen)

	# Neighbour
	G = Model(ae_model.encoder_input, z_layer)
	G.compile(optimizer='adam',
				loss='categorical_crossentropy',
				metrics=['accuracy'])
	x_input = G.predict(y_test_gen)

	for i in range(10):
		print i
		g_input = neighbours(x_input[i], diff_std, layers=2)
		gen = F.predict(g_input)
		viz.plot_number(gen)

	# generate transitions from one number to another
	for mix in range(5,0,-1):
		transition(F, x_input, mix*0.1)



# print 'Standard feature extraction'
# feature_extraction()

# print 'Classification via matching'
# classification(x_train, y_train)
print 'Classification via completion'
classification(x_train, xy_train)
# print 'Classification via completion (all)'
# all_x_train = np.concatenate((x_train, y_train), axis=0)
# all_y_train = np.concatenate((xy_train, xy_train), axis=0)
# classification(all_x_train, all_y_train)

# print 'Generation via completion'
# generation(y_train, xy_train)