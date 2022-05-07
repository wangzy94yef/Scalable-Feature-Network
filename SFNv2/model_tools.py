from time import sleep
import keras
import numpy as np
import SFN


def concat_output():
    sleep(1000)

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

'''将token和CNNs的输出结合，训练Bi-LSTM'''
import presentation_learning

presentation = {}
# for i in range(len(token_list)):
    # presentation[i] = [token_list[i], history_concat[i]]
    # presentation[i] = np.append(token_list[i], history_concat[i])
opt = keras.optimizers.SGD(lr=0.0001, decay=1e-6)
model_presentation = presentation_learning.build_presentation_learning(token_list)
model_presentation.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

lableData = SFN.lableData

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