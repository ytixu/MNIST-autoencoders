import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, Reshape, Flatten
from keras.layers import Conv2D, MaxPooling2D

class Flatten_AE():

	def __init__(self, input_size, latent_size):
		self.encoder_input = Input(shape=(input_size,), name='encoder_input')
		encoder_layer = Dense(latent_size*3, activation='relu', name='encoder_dense_1')(self.encoder_input)
		encoder_layer = Dropout(0.2, name='encoder_drop_1')(encoder_layer)
		encoder_layer = Dense(latent_size*2, activation='relu', name='encoder_dense_2')(encoder_layer)
		encoder_layer = Dropout(0.2, name='encoder_drop_2')(encoder_layer)
		self.encoder_layer = Dense(latent_size, activation='sigmoid', name='encoder_dense_3')(encoder_layer)

		decode_dense_1 = Dense(latent_size*2, activation='relu', name='decoder_dense_1')
		decode_dense_2 = Dense(latent_size*3, activation='relu', name='decoder_dense_2')
		decode_dense_3 = Dense(input_size, activation='sigmoid', name='decoder_dense_3')
		decoder_layer = decode_dense_3(decode_dense_2(decode_dense_1(self.encoder_layer)))

		self.ae = Model(self.encoder_input, decoder_layer)
		self.encoder = Model(self.encoder_input, self.encoder_layer)
		self.decoder_layers = lambda x: decode_dense_3(decode_dense_2(decode_dense_1(x)))

class CNN_AE():
	def __init__(self, input_size, latent_size, img_size):
		self.encoder_input = Input(shape=(input_size,)) 
		x = Dense(img_size**2, activation='relu')(self.encoder_input)
		x = Reshape((img_size, img_size))(x)
		x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
		x = MaxPooling2D((2, 2), padding='same')(x)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
		x = MaxPooling2D((2, 2), padding='same')(x)
		x = Conv2D(8, (3, 3), activation='sigmoid', padding='same')(x)
		x = MaxPooling2D((2, 2), padding='same')(x)
		# at this point the representation is (4, 4, 8) i.e. 128-dimensional
		
		x = Flatten()(x)
		self.encoder_layer = x #Activation('sigmoid')(x)

		dec_1 = Reshape((4,4,8))
		dec_2 = Conv2D(8, (3, 3), activation='relu', padding='same')
		dec_3 = UpSampling2D((2, 2))
		dec_4 = Conv2D(8, (3, 3), activation='relu', padding='same')
		dec_5 = UpSampling2D((2, 2))
		dec_6 = Conv2D(16, (3, 3), activation='relu')
		dec_7 = UpSampling2D((2, 2))
		dec_8 = Conv2D(1, (3, 3), activation='relu', padding='same')
		dec_9 = Flatten()
		dec_10 = Dense(input_size, activation='sigmoid')

		self.decoded_layers = lambda x: dec_10(dec_9(dec_8(dec_7(dec_6(dec_5(dec_4(dec_3(dec_2(dec_1(x))))))))))
		decoded_layer = self.decoded_layers(self.encoded_layer)

		self.ae = Model(input_img, decoded)
		self.encoder = Model(self.encoder_input, self.encoder_layer)