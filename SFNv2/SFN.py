import keras
import pandas as pd
import numpy as np

dataset = pd.read_csv("./multiscale_dataset/cms/PU_dataset_1.csv", header=None)

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
print(lineData.shape)
print(funcData.shape)
print(lableData.shape)

# ===============================token=====================================

import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate, Input
from keras.layers.recurrent import LSTM
from keras import losses, Model
from keras import optimizers
from keras import backend as K

'''
通过load_token得到embedded的token的ndarray
'''

import os

current_path1 = os.path.abspath(__file__)
print(current_path1)

# re.search('^[0-9]')

token_arr_1 = np.loadtxt(
    './multiscale_dataset/cgd/cwe119_cgd.txt',
    dtype=str,
    comments='-',
    delimiter='---------------------------------',
    encoding='UTF-8'
)

token_arr_2 = np.loadtxt(
    './multiscale_dataset/cgd/cwe399_cgd.txt',
    dtype=str,
    comments='-',
    delimiter='---------------------------------',
    encoding='UTF-8'
)

token_arr = np.append(token_arr_1, token_arr_2)
print("length of token set 1: ", len(token_arr_1))
print("length of token set 2: ", len(token_arr_2))
print("length of token set all: ", len(token_arr))
print(token_arr[6] == "0")

n = 0
token_list = []

str = ""
for t in token_arr:
    if t != "0" and t != "1":
        str += t
    else:
        # print(str)
        token_list.append(str)
        str = ""

for i in range(10):
    print(token_list[i])

import load_token
'''获取经过词嵌入的token'''
embeddedToken = load_token.getTokenEmbedded(token_list)

# =======================model build phase============================

import CNNs

model_Line = CNNs.build_model_line([10, 1])
# print(model_Line.summary())
model_func = CNNs.build_model_func([11, 1])
# print(model_func.summary())

print(model_Line.outputs.__class__)

model_concat = concatenate([model_Line.output, model_func.output])
print(model_concat.__class__)
print("class: !!! : !!!： ",lineData.__class__)
print("class: !!! : !!!： ",[lineData, funcData].__class__)

#
model_concat = Dense(128,activation="relu", kernel_initializer="uniform")(model_concat)
model_concat = Dropout(0.25)(model_concat)
model_concat = Dense(16,activation="relu", kernel_initializer="uniform")(model_concat)
model_concat = Dropout(0.25)(model_concat)
model_concat = Dense(1,activation="relu", kernel_initializer="uniform")(model_concat)
#
model_concat = Model(inputs = [model_Line.input, model_func.input], outputs = model_concat)

opt = keras.optimizers.SGD(lr=0.0001, decay=1e-6)
model_concat.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
print(model_concat.summary())
#
# model_concat = Dense(1, activation="softmax")(model_concat)
#
# model = Model(inputs = [model_Line.input,model_func.input], outputs = model_concat)

# # ==================concat train=======================

from timeit import default_timer as timer
start = timer()
history_concat = model_concat.fit([lineData, funcData],
                    lableData,
                    batch_size=32,
                    epochs=300,
                    validation_split=0.2,
                    verbose=1)
end = timer()
print("concat训练时间： ", end - start)

# # ==================build presentation model=======================

'''将token和CNNs的输出结合，训练Bi-LSTM'''
import presentation_learning

presentation = {}
for i in range(len(token_list)):
    # presentation[i] = [token_list[i], history_concat[i]]
    presentation[i] = np.append(token_list[i], history_concat[i])

model_presentation = presentation_learning.build_presentation_learning(presentation)
model_presentation.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

from timeit import default_timer as timer
start = timer()
history_presentation = model_presentation.fit(presentation,
                    lableData,
                    batch_size=16,
                    epochs=30,
                    validation_split=0.2,
                    verbose=1)
end = timer()
print("presentation训练时间： ", end - start)

# # ==================get view of results=======================

history_dict_line = history_concat.history
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

# # ==================get view of results=======================

history_dict_presentation = history_presentation.history
history_dict_presentation.keys()

import matplotlib.pyplot as plt

loss_values = history_dict_presentation['loss']
val_loss_values = history_dict_presentation['val_loss']
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
