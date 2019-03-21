import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, Reshape, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

class Flatten_AE():

	def __init__(self, input_size, latent_size):
		self.encoder_input = Input(shape=(input_size,), name='encoder_input')
		encoder_layer = Dense(latent_size*3, activation='relu', name='encoder_dense_1')(self.encoder_input)
		encoder_layer = Dropout(0.2, name='encoder_drop_1')(encoder_layer)
		encoder_layer = Dense(latent_size*2, activation='relu', name='encoder_dense_2')(encoder_layer)
		encoder_layer = Dropout(0.2, name='encoder_drop_2')(encoder_layer)
		self.encoder_layer = Dense(latent_size, activation='sigmoid', name='encoder_dense_3')(encoder_layer)

		decode_dense_1 = Dense(latent_size*2, activation='relu', name='decoder_dense_1')
		# decode_drop = Dropout(0.2, name='encoder_drop')
		decode_dense_2 = Dense(latent_size*3, activation='relu', name='decoder_dense_2')
		decode_dense_3 = Dense(input_size, activation='sigmoid', name='decoder_dense_3')
		self.decoder_layers = lambda x: decode_dense_3(decode_dense_2(decode_dense_1(x)))
		# self.decoder_layers = lambda x: decode_dense_3(x)
		decoder_layer = self.decoder_layers(self.encoder_layer)

		self.ae = Model(self.encoder_input, decoder_layer)
		self.encoder = Model(self.encoder_input, self.encoder_layer)

# class CNN_AE():
# 	def __init__(self, input_size, latent_size, img_size):
# 		self.encoder_input = Input(shape=(input_size,), name='enc_input') 
# 		x = Dense(img_size**2, activation='relu', name='enc_dense_1')(self.encoder_input)
# 		x = Dropout(0.2, name='encoder_drop')(x)
# 		x = Reshape((img_size, img_size, 1), name='enc_reshape')(x)
# 		x = Conv2D(16, (3, 3), activation='relu', padding='same', name='enc_conv_1')(x)
# 		x = MaxPooling2D((2, 2), padding='same', name='enc_pool_1')(x)
# 		# x = BatchNormalization()(x)
# 		x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv_2')(x)
# 		x = MaxPooling2D((2, 2), padding='same', name='enc_pool_2')(x)
# 		# x = BatchNormalization()(x)
# 		x = Conv2D(8, (3, 3), activation='relu', padding='same', name='enc_conv_3')(x)
# 		x = MaxPooling2D((2, 2), padding='same', name='enc_pool_3')(x)
# 		x = BatchNormalization()(x)
# 		# at this point the representation is (4, 4, 8) i.e. 128-dimensional
		
# 		x = Flatten(name='enc_flattent')(x)
# 		self.encoder_layer = Dense(latent_size, activation='sigmoid', name='enc_dense_2')(x)

# 		dec_0 = Dense(128, activation='relu', name='dec_dense_1')
# 		dec_1 = Reshape((4,4,8), name='dec_reshape')
# 		dec_2 = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_conv_1')
# 		dec_3 = UpSampling2D((2, 2), name='dec_sampling_1')
# 		dec_4 = Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_conv_2')
# 		dec_5 = UpSampling2D((2, 2), name='dec_sampling_2')
# 		dec_6 = Conv2D(16, (3, 3), activation='relu', name='dec_conv_3')
# 		dec_7 = UpSampling2D((2, 2), name='dec_sampling_3')
# 		# dec_8 = Conv2D(1, (3, 3), activation='relu', padding='same', name='dec_conv_4')
# 		dec_8 = Flatten(name='dec_flatten')
# 		dec_9 = Dense(input_size, activation='sigmoid', name='dec_dense_2')

# 		self.decoder_layers = lambda x: dec_9(dec_8(dec_7(dec_6(dec_5(dec_4(dec_3(dec_2(dec_1(dec_0(x))))))))))
# 		decoded_layer = self.decoder_layers(self.encoder_layer)

# 		self.ae = Model(self.encoder_input, decoded_layer)
# 		self.encoder = Model(self.encoder_input, self.encoder_layer)

# class CNN_AE():
# 	def __init__(self, input_size, latent_size, img_size):
# 		self.encoder_input = Input(shape=(input_size,), name='enc_input') 
# 		x = Dense(img_size**2, activation='relu', name='enc_dense_1')(self.encoder_input)
# 		x = Reshape((img_size, img_size, 1), name='enc_reshape')(x)
# 		x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=3, name='enc_conv_1')(x)
# 		x = MaxPooling2D((2, 2), padding='same', strides=2, name='enc_pool_1')(x)
# 		x = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2, name='enc_conv_2')(x)
# 		x = MaxPooling2D((2, 2), padding='same', strides=2, name='enc_pool_2')(x)
# 		# 2,2,8
		
# 		x = Flatten(name='enc_flattent')(x)
# 		self.encoder_layer = Activation('sigmoid')(x) #Dense(latent_size, activation='sigmoid', name='enc_dense_2')(x)

# 		dec_1 = Reshape((2,2,8), name='dec_reshape')
# 		dec_2 = Conv2DTranspose(16, (3, 3), activation='relu', strides=2, padding='valid', name='dec_conv_1')
# 		dec_3 = Conv2DTranspose(8, (5, 5), activation='relu', strides=3, padding='same', name='dec_conv_2')
# 		dec_4 = Conv2DTranspose(1, (2, 2), activation='relu', strides=2, padding='same', name='dec_conv_3')
# 		dec_5 = Flatten(name='dec_flatten')
# 		dec_6 = Dense(input_size, activation='sigmoid', name='dec_dense_1')

# 		self.decoder_layers = lambda x: dec_6(dec_5(dec_4(dec_3(dec_2(dec_1(x))))))
# 		decoded_layer = self.decoder_layers(self.encoder_layer)

# 		self.ae = Model(self.encoder_input, decoded_layer)
# 		self.encoder = Model(self.encoder_input, self.encoder_layer)

class CNN_AE():
	def __init__(self, input_size, latent_size, img_size):
		self.encoder_input = Input(shape=(input_size,), name='enc_input') 
		x = Dense(img_size**2, activation='relu', name='enc_dense_1')(self.encoder_input)
		x = Reshape((img_size, img_size, 1), name='enc_reshape')(x)
		x = Conv2D(32, (3, 3), activation='relu', padding='same', name='enc_conv_1')(x)
		x = Conv2D(64, (3, 3), activation='relu', padding='same', name='enc_conv_2')(x)
		x = MaxPooling2D((2, 2), padding='same', name='enc_pool_2')(x)
		# x = Dropout(0.2)(x) #0.25
		x = BatchNormalization()(x)
		x = Flatten(name='enc_flattent')(x)
		# x = Dense(3136, activation='relu')(x)
		# x = Dropout(0.2)(x) #0.25
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.2)(x) # 0.5
		self.encoder_layer = Dense(latent_size, activation='sigmoid', name='enc_dense_2')(x)

		dec_0 = Dense(128, activation='relu', name='dec_dense_1')
		# dec_0 = Dense(3136, activation='relu', name='dec_dense_2')
		dec_1 = Dense(12544, activation='relu', name='dec_dense_3')
		dec_2 = Reshape((14, 14, 64), name='dec_reshape')
		dec_3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='dec_conv_1')
		dec_4 = UpSampling2D((2, 2), name='dec_sampling_1')
		dec_5 = Conv2D(32, (3, 3), activation='relu', padding='same', name='dec_conv_2')
		dec_6 = Flatten(name='dec_flatten')
		dec_7 = Dense(input_size, activation='sigmoid', name='dec_dense_4')

		self.decoder_layers = lambda x: dec_7(dec_6(dec_5(dec_4(dec_3(dec_2(dec_1(dec_0(x))))))))
		decoded_layer = self.decoder_layers(self.encoder_layer)

		self.ae = Model(self.encoder_input, decoded_layer)
		self.encoder = Model(self.encoder_input, self.encoder_layer)
