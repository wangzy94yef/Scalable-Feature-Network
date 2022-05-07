import os

import numpy as np
import re

# get token===============================
from keras_preprocessing.text import one_hot

current_path1 = os.path.abspath(__file__)
print(current_path1)

# re.search('^[0-9]')

token_arr = np.loadtxt(
    './multiscale_dataset/cgd/cwe119_cgd.txt',
    dtype=str,
    # comments=re.search('^[0-9]'),
    comments='-',
    delimiter='---------------------------------',
    encoding='UTF-8'
)

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

# get label=======================================
import keras
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
print(dataset)

lableData = dataset[:, -1]
print(lableData)
print(amount_of_feature)
print(lableData.shape)
print("========================================================")

# embedding======================================================
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate, Input, Embedding
from keras.layers.recurrent import LSTM
from keras import losses, Model
from keras import optimizers
from keras import backend as K
from keras.models import Sequential
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths, extract_embeddings
#
# vocab_size = 50;
# encoded_token = [one_hot(t, vocab_size, filters=" ") for t in token_list]
# print(encoded_token)
# print(len(encoded_token))

model_path = get_pretrained(PretrainedList.multi_cased_base)
paths = get_checkpoint_paths(model_path)
Data = token_list
# embeddings = extract_embeddings(model_path, data)



import codecs

def getTokenEmbedded(tokenData):
    token_dict = {}
    with codecs.open(paths.vocab, 'r', encoding='utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    from keras_bert import load_trained_model_from_checkpoint
    model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint)

    from keras_bert import Tokenizer
    import numpy as np
    tokenizer = Tokenizer(token_dict)
    embedded_token_list = []
    for text in tokenData:
        indices, segments = tokenizer.encode(first=text, max_len=512)
        # print(indices[:10])
        # print(segments[:10])
        predicts = model.predict([np.array([indices]), np.array([segments])])[0]
        embedded_token_list.append(predicts)
        print(predicts[:10])
    return token_dict
