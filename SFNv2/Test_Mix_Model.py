import os
import sys
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import math

from skimage.filters import gabor_kernel

import keras
from keras import __version__, Input
from keras import regularizers
from keras import backend as K

from keras.applications.densenet import DenseNet201,preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras.models import Sequential

from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.constraints import maxnorm

from keras.optimizers import SGD
from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Activation, concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

if K.backend() == 'tensorflow':
    K.set_image_dim_ordering("tf")
from skimage.color import rgb2gray
from scipy import ndimage as ndi

# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp

# Loading the CIFAR-10 datasets
from keras.datasets import cifar10
import cv2
from keras.utils import plot_model
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle

IM_WIDTH, IM_HEIGHT = 128, 128  
FC_SIZE = 1024

  
# 数据集目录
train_dir = 'Datasets/train/'  
val_dir = 'Datasets/validation/' 

# 训练参数
nb_epoch = 2000
batch_size = 32

nb_train_samples= 612
nb_calsses = 153
nb_val_samples= 306

nb_epoch = int(nb_epoch)
batch_size = int(batch_size)


# 数据数据集
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255,
)



train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (IM_WIDTH, IM_HEIGHT),
    color_mode ='grayscale',
    classes=None,
    batch_size = batch_size,
    shuffle=False,
    class_mode = 'categorical'  
)
train_generator1 = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (IM_WIDTH, IM_HEIGHT),
    color_mode ='grayscale',
    classes=None,
    batch_size = batch_size,
    shuffle=False,
    class_mode = 'categorical'
)

validation_generator = test_datagen.flow_from_directory(
    directory = val_dir,
    target_size = (IM_WIDTH, IM_HEIGHT),
    color_mode = 'grayscale',
    classes=None,
    batch_size = batch_size,
    shuffle=False,
    class_mode = 'categorical'
)
validation_generator1 = test_datagen.flow_from_directory(
    directory = val_dir,
    target_size = (IM_WIDTH, IM_HEIGHT),
    color_mode = 'grayscale',
    classes=None,
    batch_size = batch_size,
    shuffle=False,
    class_mode = 'categorical'
)


# 定义网络模型
def Create_Cnn(width, height, depth, filters=(96, 48, 64),regress = False):
    inputShape = (height, width, depth)
    chanDim = -1

    inputs = Input(shape = inputShape)

    for (i,f) in enumerate(filters):
        if i == 0:
            x = inputs
        x = Conv2D(f, (3, 3), padding = "same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = chanDim)(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)
        x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)


    if regress:

        x = Dense(nb_calsses)(x)
        x = Activation('softmax')(x)


    model = Model(inputs, x)

    return model

Cnn_One = Create_Cnn(128, 128, 1, regress = False)
Cnn_Two = Create_Cnn(128, 128, 1, regress = False)


# 模型合并
combinedInput = concatenate([Cnn_One.output, Cnn_Two.output])

# 
x = Dense(nb_calsses, activation = "softmax")(combinedInput)

# 整合模型
model = Model(inputs = [Cnn_One.input,Cnn_Two.input], outputs = x)

# print(model.summary())

# 
opt = keras.optimizers.rmsprop(lr=0.0001, rho=0.9, epsilon=1e-6, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


def generator_Two(generator1,generator2):
    while True:
        for (x1,y1),(x2,y2) in zip(generator1,generator2):
            yield [x1,x2],y1



history_fit = model.fit_generator(
    generator_Two(train_generator,train_generator1),
    samples_per_epoch = math.ceil(nb_train_samples // batch_size),
    validation_data = generator_Two(validation_generator,validation_generator1),
    validation_steps = math.ceil(nb_val_samples // batch_size),
    nb_epoch=nb_epoch
)





'''
list1 = []

for i in train_generator.class_indices:
    list1.append(i)
list2 = np.array(list1)
print(list2)

# model.layers[0].trainable = False

'''


