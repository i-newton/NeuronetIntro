import csv
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from numpy import genfromtxt

batch_size = 128
num_classes = 10
epochs = 20


def read_training(filename):
    my_data = genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.float32)
    labels = my_data[:, 0]
    input_weights = my_data[:, 1:]
    return input_weights, labels


def read_test(filename):
    return genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.float32)

x_train, y_train = read_training('train.csv')
y_train = keras.utils.to_categorical(y_train, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

test = read_test('test.csv')
result_shape = (test.shape[0], 2)
result = np.zeros(result_shape, dtype=np.int_)
for i in range(result.shape[0]):
    array = np.array([test[i]])
    res = model.predict(array)[0]
    for detected in xrange(len(res)):
        if res[detected] > 0:
            result[i][0] = i + 1
            result[i][1] = detected
            break
    else:
        result[i][0] = i + 1
        result[i][1] = 0

np.savetxt('result.csv', result, header='ImageId,Label', fmt='%d', delimiter=',', comments='')



