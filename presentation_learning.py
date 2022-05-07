import math

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate, Input
from keras.layers.recurrent import LSTM
from keras import losses, Model
from keras import optimizers
from keras import backend as K

opt = keras.optimizers.sgd(lr=0.0001, decay=1e-6)

def build_presentation_learning(input):

    inputs = Input(shape=input)
    x = Dense(128, activation="relu", kernel_initializer="uniform")(inputs)
    x = LSTM(40, return_sequences=True)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dense(32, activation="relu", kernel_initializer="uniform")(x)
    # model.add(Dropout(0.2))
    x = Dense(1, activation="relu", kernel_initializer="uniform")(x)
    model = Model(inputs=inputs, outputs=x)
    print(model.summary())
    return model