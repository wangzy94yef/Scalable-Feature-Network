from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import tensorflow
# from tensorflow.python.keras.layers import Lambda
# from tensorflow.python.keras.layers import Lambda

import readDataset
import tensorflow as tf

from model_tools import concat_output

dataset = pd.read_csv("SFNv2/multiscale_dataset/cms/PU_dataset_1.csv", header=None)
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
# cmsData = dataset[:, :-1]
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
# cmsData = np.reshape(cmsData, (cmsData.shape[0], cmsData.shape[1], 1))
# lableData = np.reshape(lableData, (lableData.shape[0], lableData[1], 1))
print(lineData.shape)
print(funcData.shape)
print(lableData.shape)

from keras.preprocessing.sequence import TimeseriesGenerator

line_gen = TimeseriesGenerator(
    lineData,
    lableData,
    length=10,
    shuffle=False,
    reverse=False,
    batch_size=8
)
func_gen = TimeseriesGenerator(
    funcData,
    lableData,
    length=10,
    shuffle=False,
    reverse=False,
    batch_size=8
)

import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate, Lambda
from keras.layers.recurrent import LSTM
from keras import losses, Model, metrics
from keras import optimizers
from keras import __version__, Input

#定义网络模型

def expand_dim(x):
    x1 = tensorflow.expand_dims(x, axis=-1)
    return x1

# def compound_cnn(width, height, depth, filters = (24, 48), regress = False):
def compound_cnn(width, height, filters = (24, 48), regress = False):

    input_shape = (height, width)
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
    # x = Dropout(0.2)(x)

    # x = Dense(1)(x)

    model = Model(inputs, x)
    return model

CNN_line = compound_cnn(1, 10, regress=False)
CNN_func = compound_cnn(1, 11, regress=False)

print(CNN_line.__class__)

combinedCNNInput = concatenate([CNN_line.output, CNN_func.output])

# combinedCNNInput = tensorflow.expand_dims(combinedCNNInput, axis=-1)
# combinedCNNInput = Lambda(expand_dim)(combinedCNNInput)
combinedCNNInput = Lambda(lambda x : tensorflow.expand_dims(combinedCNNInput, axis=-1))(combinedCNNInput)


# print(combinedCNNInput.shape)
# print(combinedCNNInput.__class__)
x = LSTM(32, return_sequences=False)(combinedCNNInput)
x = Dense(32, activation="relu", kernel_initializer="uniform")(x)
x = Dense(1, activation = "softmax")(x)


# Bi-LSTM start...




# Bi-LSTM end...

model = Model(inputs = [CNN_line.input,CNN_func.input], outputs = x)

print(model.summary())
# 打印summary
with open('CNN_CNN_BiLSTM_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

opt = keras.optimizers.sgd(lr=0.001)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
# model.compile(loss='mse',
#               optimizer=optimizers.SGD(),
#               metrics=[metrics.Accuracy()])

from timeit import default_timer as timer
start = timer()
history_line = model.fit([lineData,
                         funcData],
                    lableData,
                    batch_size=8,
                    epochs=30,
                    validation_split=0.2,
                    verbose=1)
end = timer()
print("line训练时间： ", end - start)

# def generator_Two(generator1,generator2):
#     while True:
#         for x1, x2 in zip(generator1,generator2):
#             yield x1,x2
#
# history_sfn = model.fit_generator(
#     generator=generator_Two(line_gen, func_gen),
#     samples_per_epoch = math.ceil(65513/8 * 0.8),
#     validation_data = generator_Two(line_gen,func_gen),
#     validation_steps = math.ceil(65513/8 * 0.2),
#     nb_epoch=30
# )

