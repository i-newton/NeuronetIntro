import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, RMSprop
from keras.datasets import mnist

from numpy import genfromtxt

batch_size = 128
num_classes = 10
epochs = 25


def read_training(filename):
    my_data = genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.float32)
    labels = my_data[:, 0]
    input_weights = my_data[:, 1:]
    return input_weights, labels


def read_test(filename):
    return genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.float32)

x_train, y_train = read_training('train.csv')
y_train = keras.utils.to_categorical(y_train, num_classes)

(_, __), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(784, activation='sigmoid', input_shape=(784,)))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))

x_test = read_test('test.csv')
result_shape = (x_test.shape[0], 2)
result = np.zeros(result_shape, dtype=np.int_)
for i in range(result.shape[0]):
    array = np.array([x_test[i]])
    res = model.predict(array)[0]
    max_val = 0
    for detected in range(1, len(res)):
        if res[detected] > res[max_val]:
            max_val = detected
    result[i][0] = i + 1
    result[i][1] = max_val

np.savetxt('result.csv', result, header='ImageId,Label', fmt='%d', delimiter=',', comments='')

