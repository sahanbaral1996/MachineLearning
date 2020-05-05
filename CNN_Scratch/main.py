import numpy as np
from tensorflow import keras
from tensorflow.keras import utils
np.set_printoptions(suppress=True)
import Network
import Dense
import Activation
import loss
from Activation_functions import tanh,tanh_prime
import Conv2D
import Flatten
import cv2 as cv
import matplotlib.pyplot as plt

# load MNIST from server
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = utils.to_categorical(y_test)

neural = Network.Network()
neural.add(Conv2D.ConvLayer((28,28,1),64))
neural.add(Activation.Activation(tanh, tanh_prime))
neural.add(Conv2D.ConvLayer((26,26,64),16))
neural.add(Activation.Activation(tanh, tanh_prime))
neural.add(Flatten.Flatten())
neural.add(Dense.Dense(24*24*16, 100))
neural.add(Activation.Activation(tanh, tanh_prime))
neural.add(Dense.Dense(100, 10))
neural.add(Activation.Activation(tanh, tanh_prime))

neural.use(loss.mse, loss.mse_prime)
neural.fit(x_train[0:4000],y_train[0:4000], 15, 0.1)

print(neural.predict(x_test[0:5]))

print(y_test[0:5] )

