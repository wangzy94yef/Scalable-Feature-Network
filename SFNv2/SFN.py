import pandas as pd
import numpy as np
import readDataset

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
print("===================")

print(lineData.shape)

print("===================")

print(lineData.shape[0])
print(lineData.shape[1])
print(funcData.shape[0])
print(funcData.shape[1])
print("===================")

print(amount_of_feature)

lineData = np.reshape(lineData, (lineData.shape[0], lineData.shape[1], 1))
funcData = np.reshape(funcData, (funcData.shape[0], funcData.shape[1], 1))
# lableData = np.reshape(lableData, (lableData.shape[0], lableData[1], 1))
print(lineData.shape)
print(funcData.shape)
print(lableData.shape)

import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers

def build_model_line(input):
    model = Sequential()
    model.add(Dense(128, input_shape=(input[1], input[0])))
    model.add(Conv1D(filters=24, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters=48, kernel_size=2, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(LSTM(40, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    # model.outputs
    model.compile(loss='mse', optimizer='SGD', metrics=['acc'])

    return model

def build_model_func(input):
    model = Sequential()
    model.add(Dense(128, input_shape=(input[1], input[0])))
    model.add(Conv1D(filters=24, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters=48, kernel_size=2, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(LSTM(40, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    # model.outputs
    model.compile(loss='mse', optimizer='SGD', metrics=['acc'])

    return model


model_Line = build_model_line([1, 10, 1])
print(model_Line.summary())
model_func = build_model_func([1, 11, 1])
print(model_func.summary())

model_concat = np.concatenate([model_Line.outputs, model_func.outputs])
print(model_concat)




# ==================line train=======================

from timeit import default_timer as timer
start = timer()
history_line = model_Line.fit(lineData,
                    lableData,
                    batch_size=8,
                    epochs=30,
                    validation_split=0.2,
                    verbose=1)
end = timer()
print("line训练时间： ", end - start)


# ==================func train=======================

from timeit import default_timer as timer
start = timer()
history_func = model_func.fit(funcData,
                    lableData,
                    batch_size=8,
                    epochs=30,
                    validation_split=0.2,
                    verbose=1)
end = timer()
print("func训练时间： ", end - start)


# ==================figure of line=======================


history_dict_line = history_line.history
history_dict_line.keys()

import matplotlib.pyplot as plt

loss_values = history_dict_line['loss']
val_loss_values = history_dict_line['val_loss']
loss_values50 = loss_values[0:150]
val_loss_values50 = val_loss_values[0:150]
epochs = range(1, len(loss_values50) + 1)
plt.plot(epochs, loss_values50, 'b',color = 'blue', label='Training loss')
plt.plot(epochs, val_loss_values50, 'b',color='red', label='Validation loss')
plt.rc('font', size = 18)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks(epochs)
fig = plt.gcf()
fig.set_size_inches(15,7)
#fig.savefig('img/25/mrftest&validationlossconv1dlstm.png', dpi=300)
plt.show()

# ==================figure of func=======================

history_dict_func = history_func.history
history_dict_func.keys()

loss_values = history_dict_func['loss']
val_loss_values = history_dict_func['val_loss']
loss_values50 = loss_values[0:150]
val_loss_values50 = val_loss_values[0:150]
epochs = range(1, len(loss_values50) + 1)
plt.plot(epochs, loss_values50, 'b',color = 'blue', label='Training loss')
plt.plot(epochs, val_loss_values50, 'b',color='red', label='Validation loss')
plt.rc('font', size = 18)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks(epochs)
fig = plt.gcf()
fig.set_size_inches(15,7)
#fig.savefig('img/25/mrftest&validationlossconv1dlstm.png', dpi=300)
plt.show()
