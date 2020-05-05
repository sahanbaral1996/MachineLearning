import numpy as np
import pickle
from scipy import signal
from Layer import Layer

class ConvLayer(Layer):

    def __init__(self, input_size,num_filters, img_depth=3):
        self.num_filters = num_filters
        self.input_size = input_size
        self.filter = np.random.rand(3,3, input_size[2],num_filters)-0.5
        self.input_data = np.zeros(input_size)
        convoluted_x = input_size[0]-2
        convoluted_y = input_size[1]-2
        self.convoluted = np.zeros((convoluted_x,convoluted_y,num_filters))
        self.bias = np.random.rand(input_size[2]) - 0.5

    def forward_propagation(self, input_data):
        self.input_data = input_data
        for i in range(self.num_filters):
            for j in range(input_data.shape[2]):
                self.convoluted[:, :, i] = signal.correlate2d(input_data[:,:,j], self.filter[:, :, j, i],'valid') + self.bias[j]
        return self.convoluted

    def backward_propagation(self,output_error, learning_rate):
        in_error = np.zeros(self.input_size) #5,5,3
        w_error = np.zeros(self.filter.shape) # 3,3,3,8
        dBias = np.zeros(self.input_size[2])

        for i in range(self.input_data.shape[2]):
            for j in range(self.num_filters):

                in_error[:,:,i] += signal.convolve2d(output_error[:,:,j], self.filter[:,:,i,j], 'full')
                w_error[:, :, i, j] = signal.correlate2d(self.input_data[:, :, i], output_error[:, :, j], 'valid')

        self.filter -= learning_rate * w_error
        return in_error