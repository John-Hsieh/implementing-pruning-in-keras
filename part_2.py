import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras.datasets import cifar10
from keras.utils import np_utils

from utils import *


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
X_train /= 255
X_test /= 255
X_train=2*X_train-1
X_test=2*X_test-1


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

batch_size=100
lr=0.001
Training=True
Compressing=False

def get_model():
	batch_norm_alpha=0.9
	batch_norm_eps=1e-4

	model=Sequential()

	model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid',input_shape=[32,32,3]))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

	model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

	model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	#model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	return model

def convert_to_masked_model(model):
	#implement a function that takes a model and returns another model with masked_conv and masked_dense layers
	
	return model


model=get_model()

weights_path='pretrained_cifar10.h5'
model.load_weights(weights_path)
opt = keras.optimizers.Adam(lr=0.001,decay=1e-6)

#complie the model with sparse categorical crossentropy loss function as you did in part 1

#make sure weights are loaded correctly by evaluating the model here and printing the output


#convert the layers to maskable_layers:
prunable_model=convert_to_masked_model(model)


opt = keras.optimizers.Adam(lr=0.001,decay=1e-6)
#now complie the prunable model with sparse categorical crossentropy loss function as you did in part 1

#make sure weights are loaded correctly by evaluating the prunable model here and printing the output


#do the rest of the project:

