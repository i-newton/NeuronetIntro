import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,MaxPooling2D, Flatten
from keras import backend
from numpy import genfromtxt

batch_size = 128
num_classes = 10
epochs = 15


def read_training(filename):
    my_data = genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.float32)
    labels = my_data[:, 0]
    input_weights = my_data[:, 1:]
    return input_weights, labels


def read_test(filename):
    return genfromtxt(filename, delimiter=',', skip_header=1, dtype=np.float32)

x_train, y_train = read_training('train.csv')
y_train = keras.utils.to_categorical(y_train, num_classes)


img_rows, img_cols = 28, 28

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
"""
model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5),
                 activation='sigmoid',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
"""
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

x_test = read_test('test.csv')
if backend.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

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

