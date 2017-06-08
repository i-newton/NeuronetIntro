from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
import math
import random
from matplotlib import pyplot


def get_inputs(sample_num=1000):
    step = 2*math.pi/sample_num
    begin = - math.pi
    return [begin + i*step for i in range(sample_num)]


def get_outputs(inputs):
    return [math.cos(x) for x in inputs]


def get_test_inputs(sample_num=1000):
    begin = - math.pi
    step = 2*math.pi / sample_num
    return [begin + random.randint(0, sample_num) * step for i in range(sample_num)]


# specify model
model = Sequential()
model.add(Dense(30, input_dim=1, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
optm = SGD(lr=0.1, momentum=0.1)

model.compile(optimizer=optm, loss='mean_squared_error', metrics=['accuracy'])

# prepare training data
inputs = get_inputs(1000)
outputs = get_outputs(inputs)

test_inputs = get_test_inputs(1000)
test_outputs = get_outputs(test_inputs)

# train the model
model.fit(inputs, outputs, epochs=500, batch_size=100, verbose=1, validation_data=(test_inputs, test_outputs))

# evaluate the model
test_inputs = get_test_inputs(50)
test_outputs = get_outputs(test_inputs)
model.evaluate(test_inputs, test_outputs)

out = model.predict(inputs)
pyplot.plot(inputs, out)
pyplot.show()

pyplot.plot(inputs, outputs)
pyplot.show()
