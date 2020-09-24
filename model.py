import tensorflow as tf #google framework to build DL and ML model
import keras
from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D,Conv2DTranspose
from keras.callbacks import History
autoencoder = tf.keras.models.Sequential()
#encoder
autoencoder.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 5, strides = 1,activation='relu', padding = 'same', input_shape = (512,512,1)))
autoencoder.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 5, strides = 1, activation='relu',padding='same'))
autoencoder.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 5, strides = 1, activation='relu', padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 5, strides = 1, activation='relu', padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 5, strides = 1, activation='relu', padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2D(filters=15, kernel_size = 5, strides = 1, activation='relu'))
#decoder
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=15, kernel_size = 5, strides = 1, activation='relu'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=15, kernel_size = 5, strides = 1, activation='relu', padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=15, kernel_size = 5, strides= 1, activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=15, kernel_size = 5, strides = 1, activation = 'relu', padding = 'same'))
autoencoder.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size = 5, strides = 1, activation = 'relu', padding = 'same'))
