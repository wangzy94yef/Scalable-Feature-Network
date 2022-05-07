
import math

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate, Input
from keras.layers.recurrent import LSTM
from keras import losses, Model
from keras import optimizers
from keras import backend as K

opt = keras.optimizers.sgd(lr=0.0001, decay=1e-6)

def build_model_line(input):

    inputs = Input(shape=input)
    x = Dense(128, activation="relu", kernel_initializer="uniform")(inputs)
    x = Conv1D(filters=24, kernel_size=1, padding='same', activation='relu', kernel_initializer="uniform")(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=48, kernel_size=2, padding='same', activation='relu', kernel_initializer="uniform")(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = LSTM(40, return_sequences=True)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dense(32, activation="relu", kernel_initializer="uniform")(x)
    # model.add(Dropout(0.2))
    x = Dense(1, activation="relu", kernel_initializer="uniform")(x)
    model = Model(inputs=inputs, outputs=x)

    '''
    model = Sequential()
    model.add(Dense(128, input_shape=(input[1], input[0])))
    model.add(Conv1D(filters=24, kernel_size=1, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Conv1D(filters=48, kernel_size=2, padding='valid', activation='relu', kernel_initializer="uniform"))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(LSTM(40, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(32, activation="relu", kernel_initializer="uniform"))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation="relu", kernel_initializer="uniform"))
    # model.outputs
    model.compile(loss='mse', optimizer=opt, metrics=['acc'])
    '''
    return model

def build_model_func(input):
    inputs = Input(shape=input)
    x = Dense(128, activation="relu", kernel_initializer="uniform")(inputs)
    x = Conv1D(filters=24, kernel_size=1, padding='same', activation='relu', kernel_initializer="uniform")(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=48, kernel_size=2, padding='same', activation='relu', kernel_initializer="uniform")(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = LSTM(40, return_sequences=True)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dense(32, activation="relu", kernel_initializer="uniform")(x)
    # model.add(Dropout(0.2))
    x = Dense(1, activation="relu", kernel_initializer="uniform")(x)
    model = Model(inputs=inputs, outputs=x)

    '''
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
    model.compile(loss='mse', optimizer=opt, metrics=['acc'])
    '''

    return model
