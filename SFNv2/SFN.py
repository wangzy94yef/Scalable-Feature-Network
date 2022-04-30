import pandas as pd
import numpy as np
import readDataset

dataset = pd.read_csv("./PU_dataset_1.csv", header=None)
# print(dataset.head())
# print(dataset.tail())

amount_of_feature = len(dataset.columns)

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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers

def build_model_line(input):
    model = Sequential()
    model.add(Dense(128, input_shape=(input[1], input[0])))
    model.add(Conv1D(filters=24, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters=48, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(LSTM(40, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def build_model_func(input):
    model = Sequential()
    model.add(Dense(128, input_shape=(input[1], input[0])))
    model.add(Conv1D(filters=24, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters=48, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(LSTM(40, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

model_Line = build_model_line([1, 10, 1])
print(model_Line.summary())
model_func = build_model_func([1, 11, 1])
print(model_func.summary())

# ==================line train=======================

from timeit import default_timer as timer
start = timer()
history = model_Line.fit(lineData,
                    lableData,
                    batch_size=128,
                    epochs=100,
                    validation_split=0.2,
                    verbose=2)
end = timer()
print(end - start)
history_dict = history.history
history_dict.keys()

# ==================func train=======================

from timeit import default_timer as timer
start = timer()
history = model_func.fit(funcData,
                    lableData,
                    batch_size=128,
                    epochs=100,
                    validation_split=0.2,
                    verbose=2)
end = timer()
print(end - start)
history_dict = history.history
history_dict.keys()

# ==================ending of train...=======================