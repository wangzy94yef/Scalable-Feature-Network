import pandas as pd
import numpy as np
import readDataset
import tensorflow as tf

dataset = pd.read_csv("./PU_dataset_1.csv", header=None)
# print(dataset.head())
# print(dataset.tail())

amount_of_feature = len(dataset.columns)

# 添加过采样

dataset = dataset.values
print("values")
print(dataset)

# dt = []
# sequence_length = 22
# for index in range(len(dataset) - sequence_length):
#     dt.append(dataset[index : index + sequence_length])
# print(dt)
#

dataset = np.array(dataset)
print("array")
print(dataset)

lineData = dataset[:, :10]
funcData = dataset[:, 10:-1]
lableData = dataset[:, -1]
print(lineData)
print(funcData)
print(lableData)

# print(lineData.shape[0])
# print(lineData.shape[1])
# print(funcData.shape[0])
# print(funcData.shape[1])

print(amount_of_feature)

lineData = np.reshape(lineData, (lineData.shape[0], lineData.shape[1], 1))
funcData = np.reshape(funcData, (funcData.shape[0], funcData.shape[1], 1))
# lableData = np.reshape(lableData, (lableData.shape[0], lableData[1], 1))
print(lineData.shape)
print(funcData.shape)
print(lableData.shape)

import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate
from keras.layers.recurrent import LSTM
from keras import losses, Model
from keras import optimizers
from keras import __version__, Input

#定义网络模型

input_line = tf.keras.layers.Input(shape=(1, 10))

# def compound_cnn(width, height, depth, filters = (24, 48), regress = False):
def compound_cnn(width, height, filters = (24, 48), regress = False):
# def compound_cnn(height, depth, filters = (24, 48), regress = False):
    input_shape = (height, width)
    chanDim = -1
    inputs = Input(shape=input_shape)

    for(i, f) in enumerate(filters):
        print(i)
        print(f)
        if i == 0:
            x = inputs
        x = Dense(128)(x)
        print(x.shape)
        # dense_1 (Dense)              (None, 10, 128)           256
        x = Conv1D(f, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling1D(pool_size=2, padding='valid')(x)
        # x = Conv1D(f, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform')(x)
        # x = MaxPooling1D(pool_size=2, padding='valid')(x)
        # x = LSTM(40, return_sequences=True)(x)
        # x = LSTM(32, return_sequences=False)(x)
        # x = Dense(32)(x)
        # x = Dropout(0.2)(x)
        # x = Dense(1)(x)
    x = Flatten()(x)

    x = Dense(32)(x)
    x = Dropout(0.2)(x)

    x = Dense(1)(x)

    model = Model(inputs, x)
    return model

# CNN_line = compound_cnn(1, 10, 1, regress=False)
# CNN_func = compound_cnn(1, 11, 1, regress=False)
CNN_line = compound_cnn(1, 10, regress=False)
CNN_func = compound_cnn(1, 11, regress=False)

combinedCNNInput = concatenate([CNN_line.output, CNN_func.output])

x = Dense(22, activation = "softmax")(combinedCNNInput)

model = Model(inputs = [CNN_line.input,CNN_func.input], outputs = x)

print(model.summary())

opt = keras.optimizers.rmsprop(lr=0.0001, rho=0.9, epsilon=1e-6, decay=1e-6)
model.compile(loss='mse', optimizer='SGD', metrics=['accuracy'])


