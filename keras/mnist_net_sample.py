import csv
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batch_size = 128
num_classes = 10
epochs = 20


def read_training(filename):
    input_weights = []
    labels = []
    with open(filename, 'rb') as f:
        train_set = csv.reader(f)
        for picture in train_set:
            try:
                labels.append(int(picture[0]))
                weights = [float(pixel)/255 for pixel in picture[1:]]
                input_weights.append(weights)
            except ValueError:
                continue
    return input_weights, labels


def read_test(filename):
    input_weights = []
    with open(filename, 'rb') as f:
        train_set = csv.reader(f)
        for picture in train_set:
            try:
                weights = [float(pixel)/255 for pixel in picture]
                input_weights.append(weights)
            except ValueError:
                continue
    return input_weights

x_train, y_train = read_training('train.csv')
# convert class vectors to binary class matrices
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

x_test = read_test('test.csv')
with open('result.csv', 'w') as out:
    out.write('ImageId, Label\n')
    for i, tst in enumerate(x_test):
        out.write(str(i))
        out.write(',')
        res = model.predict(np.array([tst]))[0]
        for j in xrange(len(res)):
            if res[j] > 0:
                out.write(str(j))
                break
        else:
            out.write('0')
        out.write('\n')




