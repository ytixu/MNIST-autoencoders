import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, Reshape, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, add
import keras.backend as K
import tensorflow as tf

class Flatten_AE():
	# similar to the Deep autoencoder in https://blog.keras.io/building-autoencoders-in-keras.html
	def __init__(self, input_size, latent_size):
		self.latent_activation = 'sigmoid'

		self.encoder_input = Input(shape=(input_size,), name='encoder_input')
		encoder_layer = Dense(latent_size*3, activation='relu', name='encoder_dense_1')(self.encoder_input)
		encoder_layer = Dropout(0.2, name='encoder_drop_1')(encoder_layer)
		encoder_layer = Dense(latent_size*2, activation='relu', name='encoder_dense_2')(encoder_layer)
		encoder_layer = Dropout(0.2, name='encoder_drop_2')(encoder_layer)
		self.encoder_layer = Dense(latent_size, activation=self.latent_activation, name='encoder_dense_3')(encoder_layer)

		decode_dense_1 = Dense(latent_size*2, activation='relu', name='decoder_dense_1')
		decode_dense_2 = Dense(latent_size*3, activation='relu', name='decoder_dense_2')
		decode_dense_3 = Dense(input_size, activation='sigmoid', name='decoder_dense_3')
		self.decoder_layers = lambda x: decode_dense_3(decode_dense_2(decode_dense_1(x)))
		decoder_layer = self.decoder_layers(self.encoder_layer)

		self.ae = Model(self.encoder_input, decoder_layer)
		self.encoder = Model(self.encoder_input, self.encoder_layer)

class Dense_CNN_AE():
	# similar to https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
	def __init__(self, input_size, latent_size, img_size):
		self.latent_activation = 'sigmoid'

		self.encoder_input = Input(shape=(input_size,), name='enc_input') 
		x = Dense(img_size**2, activation='relu', name='enc_dense_1')(self.encoder_input)
		x = Reshape((img_size, img_size, 1), name='enc_reshape')(x)
		x = Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv_1')(x)
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv_2')(x)
		x = MaxPooling2D((2, 2), padding='same', name='enc_pool_2')(x)
		x = BatchNormalization()(x)
		x = Flatten(name='enc_flattent')(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.2)(x) 
		self.encoder_layer = Dense(latent_size, activation=self.latent_activation, name='enc_dense_2')(x)

		dec_1 = Dense(128, activation='relu', name='dec_dense_1')
		dec_2 = Dense(12544, activation='relu', name='dec_dense_3')
		dec_3 = Reshape((14, 14, 64), name='dec_reshape')
		dec_4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv_1')
		dec_5 = UpSampling2D((2, 2), name='dec_sampling_1')
		dec_6 = Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv_2')
		dec_7 = Flatten(name='dec_flatten')
		dec_8 = Dense(input_size, activation='sigmoid', name='dec_dense_4')

		self.decoder_layers = lambda x: dec_8(dec_7(dec_6(dec_5(dec_4(dec_3(dec_2(dec_1(x))))))))
		decoded_layer = self.decoder_layers(self.encoder_layer)

		self.ae = Model(self.encoder_input, decoded_layer)
		self.encoder = Model(self.encoder_input, self.encoder_layer)
		print self.ae.summary()

class CNN_AE():
	def __init__(self, input_size, latent_size, img_size):
		self.latent_activation = 'sigmoid'
		image2d_size = img_size**2
		reshape = Reshape((img_size, img_size, 1))

		self.encoder_input = Input(shape=(input_size,), name='enc_input') 
		labels = reshape(Dense(image2d_size, activation='relu', name='enc_dense_1')(self.encoder_input))
		labels = Dropout(0.2)(labels)
		digits = reshape(Lambda(lambda x: x[:,:image2d_size])(self.encoder_input))
		x = concatenate([digits, labels], axis=-1)

		x = Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv_1')(x)
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv_2')(x)
		x = MaxPooling2D((2, 2), padding='same', name='enc_pool_2')(x)
		x = BatchNormalization()(x)
		x = Flatten(name='enc_flattent')(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.2)(x) 
		self.encoder_layer = Dense(latent_size, activation=self.latent_activation, name='enc_dense_2')(x)

		dec_1 = Dense(128, activation='relu', name='dec_dense_1')
		dec_2 = Dense(12544, activation='relu', name='dec_dense_3')
		dec_3 = Reshape((14, 14, 64), name='dec_reshape')
		dec_4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv_1')
		dec_5 = UpSampling2D((2, 2), name='dec_sampling_1')
		dec_6 = Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv_2')
		dec_7 = Flatten(name='dec_flatten')
		dec_8 = Dense(input_size, activation='sigmoid', name='dec_dense_4')

		self.decoder_layers = lambda x: dec_8(dec_7(dec_6(dec_5(dec_4(dec_3(dec_2(dec_1(x))))))))
		decoded_layer = self.decoder_layers(self.encoder_layer)

		self.ae = Model(self.encoder_input, decoded_layer)
		self.encoder = Model(self.encoder_input, self.encoder_layer)
		print self.ae.summary()

class CNN_AEs():
	def __init__(self, input_size, latent_size, img_size):
		self.latent_activation = 'sigmoid'
		reshape = Reshape((img_size, img_size, 1))

		self.encoder_input = Input(shape=(input_size,), name='enc_input')
		labels = reshape(Dense(img_size**2, activation='relu', name='enc_dense_1')(self.encoder_input))
		labels = Dropout(0.2)(labels)
		digits = reshape(Lambda(lambda x: x[:,:-10])(self.encoder_input))
		x = concatenate([digits, labels], axis=-1)

		x = Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv_1')(x)
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv_2')(x)
		x = MaxPooling2D((2, 2), padding='same', name='enc_pool_2')(x)
		x = BatchNormalization()(x)
		x = Flatten(name='enc_flattent')(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.2)(x)
		self.encoder_layer = Dense(latent_size, activation=self.latent_activation, name='enc_dense_2')(x)

		dec_1 = Dense(128, activation='relu', name='dec_dense_1')
		dec_2 = Dense(12544, activation='relu', name='dec_dense_3')
		dec_3 = Reshape((14, 14, 64), name='dec_reshape')
		dec_4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv_1')
		dec_5 = UpSampling2D((2, 2), name='dec_sampling_1')
		dec_6 = Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv_2')
		dec_7 = Flatten(name='dec_flatten')
		dec_8 = Dense(input_size, activation='sigmoid', name='dec_dense_4')
		dec_9 = Lambda(lambda x: K.concatenate([x[:,:-10], tf.nn.softmax(x[:,-10:])], axis=-1))

		self.decoder_layers = lambda x: dec_9(dec_8(dec_7(dec_6(dec_5(dec_4(dec_3(dec_2(dec_1(x)))))))))
		decoded_layer = self.decoder_layers(self.encoder_layer)

		self.ae = Model(self.encoder_input, decoded_layer)
		self.encoder = Model(self.encoder_input, self.encoder_layer)
		print self.ae.summary()

		def custum_loss(yTrue, yPred):
			return K.mean(K.binary_crossentropy(yTrue[:,:-10], yPred[:,:-10])) + K.mean(K.binary_crossentropy(yTrue[:,-10:], yPred[:,-10:]))

		self.ae.compile(optimizer='adam',
                               loss=custum_loss,
                               metrics=['mae'])

