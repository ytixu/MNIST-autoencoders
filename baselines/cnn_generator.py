import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Conv2D, UpSampling2D
from keras.utils import to_categorical

(y_train, x_train),(y_test, x_test) = mnist.load_data()
y_train, y_test = y_train / 255.0, y_test / 255.0
x_train = to_categorical(x_train)
x_test = to_categorical(x_test)

model = Sequential([
    Dense(128, activation='relu', name='dec_dense_1', input_shape=(10,)),
    Reshape((4,4,8), name='dec_reshape'),
    Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_conv_1'),
    UpSampling2D((2, 2), name='dec_sampling_1'),
    Conv2D(8, (3, 3), activation='relu', padding='same', name='dec_conv_2'),
    UpSampling2D((2, 2), name='dec_sampling_2'),
    Conv2D(16, (3, 3), activation='relu', name='dec_conv_3'),
    UpSampling2D((2, 2), name='dec_sampling_3'),
    Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='dec_conv_4'),
    Reshape((28,28))
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['mae'])

model.fit(x_train, y_train, epochs=10)

score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]

x_test_gen = np.eye(10)
gen = model.predict(x_test_gen)

def plot(gen):
    num_rows = 1
    num_cols = 10
    f, ax = plt.subplots(num_rows, num_cols, figsize=(10,1),
                        gridspec_kw={'wspace':0.03, 'hspace':0.01}, 
                        squeeze=True)
    for r in range(num_rows):
        for c in range(num_cols):
            image_index = r * num_cols + c
            ax[c].axis("off")
            ax[c].imshow(gen[image_index], cmap='gray')
            # ax[c].set_title('No. %d' % image_index)
    plt.show()
    plt.close()

plot(gen)
