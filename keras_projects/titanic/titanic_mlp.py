import csv
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

batch_size = 128
epochs = 100
num_features = 18


def convert_to_feature_vec(row_dict):
    pass_no = int(row_dict['PassengerId'])
    if 'Survived' in row_dict:
        label = int(row_dict['Survived'])
    else:
        label = None
    features = np.zeros(shape=(num_features,), dtype=np.float32)
    pclass = int(row_dict['Pclass'])
    features[pclass - 1] = 1
    sex = row_dict['Sex']
    if sex == 'male':
        features[3] = 1
    elif sex == 'female':
        features[4] = 1
    else:
        raise Exception()
    features[5] = float(row_dict['Age']) if row_dict['Age'] else 0
    features[6] = float(row_dict['SibSp']) if row_dict['SibSp'] else 0
    features[7] = float(row_dict['Parch']) if row_dict['Parch'] else 0
    features[8] = float(row_dict['Fare']) if row_dict['Fare'] else 0
    dest = row_dict['Embarked']
    if dest == 'C':
        features[9] = 1.0
    elif dest == 'Q':
        features[10] = 1.0
    elif dest == 'S':
        features[11] = 1.0
    allowed_cabin_starts = ['A', 'B', 'C', 'D', 'E', 'F']
    cabin = row_dict['Cabin'] or ''
    for i, c in enumerate(allowed_cabin_starts):
        if cabin.startswith(c):
            features[12+i] = 1.0
    return pass_no, features, label


def read_data(filename, diff=1):
    with open(filename, 'rb') as f:
        feature_matrix = np.zeros(shape=(891, num_features), dtype=np.float32)
        labels = np.zeros(shape=(891, 1))
        reader = csv.DictReader(f)
        for row in reader:
            pass_no, pass_features, label = convert_to_feature_vec(row)
            feature_matrix[pass_no-diff] = pass_features
            labels[pass_no-diff] = label
        return labels, feature_matrix

y_train, x_train, = read_data('train.csv')
y_train = np_utils.to_categorical(y_train, 2)
# train net
model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape=(num_features,)))
#model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs)
# test net
_, x_test = read_data('test.csv', 892)
x_test = x_test[:418]
result_shape = (x_test.shape[0], 2)
result = np.zeros(result_shape, dtype=np.int_)
for i in range(result.shape[0]):
    array = np.array([x_test[i]])
    res = model.predict(array)[0]
    max_val = 0
    for detected in range(1, len(res)):
        if res[detected] > res[max_val]:
            max_val = detected
    result[i][0] = i + 892
    result[i][1] = max_val

np.savetxt('result.csv', result, header='PassengerId,Survived', fmt='%d', delimiter=',', comments='')
