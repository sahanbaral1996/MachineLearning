import numpy as np
import Layer


class Dense(Layer.Layer):

    def __init__(self, input_size, output_size):
        self.weight = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1,output_size) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(input, self.weight) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weight.T)
        self.input = self.input.reshape((1,-1))
        weight_error = np.dot(self.input.T, output_error)
        self.bias -= learning_rate * output_error
        self.weight -= learning_rate * weight_error
        return input_error
