from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import MaxPooling2D

from tensorflow.examples.tutorials.mnist import input_data

def dataLoaderConcise(path=None):
    if path is None:
        path = '/home/zhaok14/example/PycharmProjects/cnn_scratch/MNIST_data'
    else:
        pass
    # load mnist data
    return input_data.read_data_sets(path, one_hot=True)

mnist = dataLoaderConcise(path=None)
x_train = mnist.train.images
y_train = mnist.train.labels
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((55000, 28, 28, 1))

# x_train = x_train.reshape((60000, 28, 28, 1))[0:2000]
# y_train = to_categorical(y_train, 10)[0:2000]

import keras
from keras.layers import Dense, Input, Conv2D, Flatten, concatenate
from keras.models import Model

def naive_inception(inputs):

    towerOne = Conv2D(6, (1,1), activation='relu', border_mode='same')(inputs)
    towerTwo = Conv2D(6, (3,3), activation='relu', border_mode='same')(inputs)
    towerThree = Conv2D(6, (5,5), activation='relu', border_mode='same')(inputs)
    x = concatenate([towerOne, towerTwo, towerThree], axis=3)
    return x

def dimension_reduction_inception(inputs):
    tower_one = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_one = Conv2D(6, (1,1), activation='relu', border_mode='same')(tower_one)

    tower_two = Conv2D(6, (1,1), activation='relu', border_mode='same')(inputs)
    tower_two = Conv2D(6, (3,3), activation='relu', border_mode='same')(tower_two)

    tower_three = Conv2D(6, (1,1), activation='relu', border_mode='same')(inputs)
    tower_three = Conv2D(6, (5,5), activation='relu', border_mode='same')(tower_three)
    x = concatenate([tower_one, tower_two, tower_three], axis=3)
    return x


def naive_model(x_train):

    inputs = Input(x_train.shape[1:])

    x = naive_inception(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)

    model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(lr=0.001),
                 metrics=['accuracy'])
    return model

modelA = naive_model(x_train)
modelA.fit(x_train, y_train, epochs=50, shuffle=True,  validation_split=0.1)

# matplotlib inline
import matplotlib.pyplot as plt
def show_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.show()

show_history(modelA.history)