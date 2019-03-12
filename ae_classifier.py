import numpy as np
import matplotlib.pyplot as plt

import utils.parser as parser
# parse args
args = parser.parse()

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, Reshape

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
BATCH_SIZE = 64

##############
# Autoencoder
if args['model'] == 'flatten':
	ae_model = Flatten_AE(INPUT_SIZE, LATENT_SIZE)
	EPOCHS = 50
	BATCH_SIZE = 64
elif args['model'] == 'cnn':
	ae_model = CNN_AE(INPUT_SIZE, LATENT_SIZE, IMG_SIZE)
	EPOCHS = 10
	BATCH_SIZE = 128

ae_model.ae.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['mae'])

ae_model.ae.fit(x_xy_y_train, x_xy_y_train, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE)

score = ae_model.ae.evaluate(x_test, x_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]



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

	ext.fit(ext_train, y_train_orig, epochs=5, shuffle=True, batch_size=16)

	score = ext.evaluate(ext_test, y_test_orig, verbose=0)
	print 'Test loss:', score[0]
	print 'Test accuracy:', score[1]

print 'Standard feature extraction'
feature_extraction()




##############
print 'Completion function'
def get_complete_func(x_train_, y_train_):

	a_train = ae_model.encoder.predict(x_train_)
	b_train = ae_model.encoder.predict(y_train_)

	# diff = b_train - a_train
	# print np.mean(diff, axis=0), np.std(diff, axis=0)

	g_input = Input(shape=(LATENT_SIZE,), name='g_input')
	g_dense = Dense(LATENT_SIZE, activation='sigmoid', use_bias=False, name='g_dense')
	g_layer = g_dense(g_input)

	g = Model(g_input, g_layer)

	g.compile(optimizer='adam',
					loss='mean_squared_error',
					metrics=['mae'])

	g.fit(a_train, b_train, epochs=5, shuffle=True, batch_size=16)

	g_layer_ = g_dense(ae_model.encoder_layer)
	decoder_layer_ = ae_model.decoder_layers(g_layer_)
	
	return decoder_layer_



##############
def classification(x_train_, xy_train_):
	class_layer = get_complete_func(x_train_, xy_train_)

	drop_x = Lambda(lambda x: x[:,-10:], output_shape=(10,), name='drop_lambda')(class_layer)
	class_output = Activation('softmax', name='softmax')(drop_x)

	F = Model(ae_model.encoder_input, class_output)
	F.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	score = F.evaluate(x_test, y_test_orig, verbose=0)
	print 'Test loss:', score[0]
	print 'Test accuracy:', score[1]

# print 'Classification via matching'
# classification(x_train, y_train)
print 'Classification via completion'
classification(x_train, xy_train)
# print 'Classification via completion (all)'
# all_x_train = np.concatenate((x_train, y_train), axis=0)
# all_y_train = np.concatenate((xy_train, xy_train), axis=0)
# classification(all_x_train, all_y_train)


##############
print 'Generation'

gen_layer = get_complete_func(y_train, xy_train)
drop_y = Lambda(lambda x: x[:,:-10], output_shape=(IMG_SIZE**2,), name='drop_lambda')(gen_layer)
gen_output = Reshape((IMG_SIZE, IMG_SIZE))(drop_y)

F = Model(ae_model.encoder_input, gen_output)
F.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

gen = F.predict(y_test_gen)

##############
# Visualize

viz.plot(gen)