import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, Reshape

(x_train_orig, y_train_orig),(x_test_orig, y_test_orig) = mnist.load_data()
x_train_orig, x_test_orig = x_train_orig / 255.0, x_test_orig / 255.0

IMG_SIZE = x_train_orig.shape[1]
LATENT_SIZE = 32

# construct complete data

def const_complete_data(x=[], y=[]):
	if (len(x) > 0):
		x = np.reshape(x, (x.shape[0], -1))
	else:
		x = np.ones((y.shape[0], IMG_SIZE**2))
	if (len(y) > 0):
		y = to_categorical(y)
	else:
		y = np.ones((x.shape[0], 10))
	return np.concatenate((x, y), axis=1)

xy_train = const_complete_data(x_train_orig, y_train_orig)
INPUT_SIZE = xy_train.shape[1]

x_train = const_complete_data(x_train_orig)
x_xy_train = np.concatenate((x_train, xy_train), axis=0)
y_train = const_complete_data([], y_train_orig)

y_test_gen = np.ones((10,INPUT_SIZE))
y_test_gen[:,-10:] = np.eye(10)

x_xy_y_train = np.concatenate((x_xy_train, y_train), axis=0)
# xy_test = const_complete_data(x_test, y_test_orig)
x_test = const_complete_data(x_test_orig)



##############
# Autoencoder
encoder_input = Input(shape=(INPUT_SIZE,), name='encoder_input')
encoder_layer = Dense(LATENT_SIZE*3, activation='relu', name='encoder_dense_1')(encoder_input)
encoder_layer = Dropout(0.2, name='encoder_drop_1')(encoder_layer)
encoder_layer = Dense(LATENT_SIZE*2, activation='relu', name='encoder_dense_2')(encoder_input)
encoder_layer = Dropout(0.2, name='encoder_drop_2')(encoder_layer)
encoder_layer = Dense(LATENT_SIZE, activation='sigmoid', name='encoder_dense_3')(encoder_input)

decode_dense_1 = Dense(LATENT_SIZE*2, activation='relu', name='decoder_dense_1')
decode_dense_2 = Dense(LATENT_SIZE*3, activation='relu', name='decoder_dense_2')
decode_dense_3 = Dense(INPUT_SIZE, activation='sigmoid', name='decoder_dense_3')
decoder_layer = decode_dense_3(decode_dense_2(decode_dense_1(encoder_layer)))

ae = Model(encoder_input, decoder_layer)
encoder = Model(encoder_input, encoder_layer)

ae.compile(optimizer='adam',
				loss='binary_crossentropy',
				metrics=['mae'])

ae.fit(x_xy_y_train, x_xy_y_train, epochs=10, shuffle=True, batch_size=64)

score = ae.evaluate(x_test, x_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]



##############
def feature_extraction():
	ext_train = encoder.predict(x_train)
	ext_test = encoder.predict(x_test)

	ext_input = Input(shape=(LATENT_SIZE,), name='ext_input')
	ext_output = Dense(10, activation='softmax')(ext_input)

	ext = Model(ext_input, ext_output)
	ext.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	ext.fit(ext_train, y_train_orig, epochs=10, shuffle=True, batch_size=16)

	score = ext.evaluate(ext_test, y_test_orig, verbose=0)
	print 'Test loss:', score[0]
	print 'Test accuracy:', score[1]

print 'Standard feature extraction'
feature_extraction()




##############
print 'Completion function'
def get_complete_func(x_train_, xy_train_):

	a_train = encoder.predict(x_train_)
	b_train = encoder.predict(xy_train_)

	diff = b_train - a_train
	print np.mean(diff, axis=0), np.std(diff, axis=0)

	g_input = Input(shape=(LATENT_SIZE,), name='g_input')
	g_dense = Dense(LATENT_SIZE, activation='sigmoid', use_bias=False, name='g_dense')
	g_layer = g_dense(g_input)

	g = Model(g_input, g_layer)

	g.compile(optimizer='adam',
					loss='mean_squared_error',
					metrics=['mae'])

	g.fit(a_train, b_train, epochs=10, shuffle=True, batch_size=16)

	g_layer_ = g_dense(encoder_layer)
	decoder_layer_ = decode_dense_3(decode_dense_2(decode_dense_1(g_layer_)))
	
	return decoder_layer_



##############
def classification(x_train_, xy_train_):
	class_layer = get_complete_func(x_train_, xy_train_)

	drop_x = Lambda(lambda x: x[:,-10:], output_shape=(10,), name='drop_lambda')(class_layer)
	class_output = Activation('softmax', name='softmax')(drop_x)

	F = Model(encoder_input, class_output)
	F.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])

	score = F.evaluate(x_test, y_test_orig, verbose=0)
	print 'Test loss:', score[0]
	print 'Test accuracy:', score[1]

print 'Classification via matching'
classification(x_train, y_train)
print 'Classification via completion'
classification(x_train, xy_train)
print 'Classification via completion (all)'
all_x_train = np.concatenate((x_train, y_train), axis=0)
all_y_train = np.concatenate((xy_train, xy_train), axis=0)
classification(all_x_train, all_y_train)


##############
print 'Generation'
gen_layer = get_complete_func(y_train, xy_train)
drop_y = Lambda(lambda x: x[:,:-10], output_shape=(IMG_SIZE**2,), name='drop_lambda')(gen_layer)
gen_output = Reshape((IMG_SIZE, IMG_SIZE))(drop_y)

F = Model(encoder_input, gen_output)
F.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

gen = F.predict(y_test_gen)



##############
# Visualize

num_rows = 2
num_cols = 5
f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),
					gridspec_kw={'wspace':0.03, 'hspace':0.1}, 
					squeeze=True)
for r in range(num_rows):
	for c in range(num_cols):
		image_index = r * num_cols + c
		ax[r,c].axis("off")
		ax[r,c].imshow(gen[image_index], cmap='gray')
		ax[r,c].set_title('No. %d' % image_index)
plt.show()
plt.close()