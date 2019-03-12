import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.utils import to_categorical

(y_train, x_train),(y_test, x_test) = mnist.load_data()
y_train, y_test = y_train / 255.0, y_test / 255.0
x_train = to_categorical(x_train)
x_test = to_categorical(x_test)

model = Sequential([
    # Dense(128, activation='relu', input_shape=(10,)),
    Dense(512, activation='relu', input_shape=(10,)),
    Dropout(0.2),
    Dense(28*28, activation='sigmoid'),
    Reshape((28, 28))
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['mae'])

model.fit(x_train, y_train, epochs=5)

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
