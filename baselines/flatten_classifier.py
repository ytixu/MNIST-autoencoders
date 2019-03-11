import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# same model as in https://www.tensorflow.org/tutorials

model = Sequential([
  Flatten(input_shape=(28, 28)),
  Dense(512, activation='relu'),
  Dropout(0.2),
  Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]

# Test loss: 0.06789465941194212
# Test accuracy: 0.9797
