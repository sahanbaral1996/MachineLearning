import numpy as np
from Layer import Layer


class Flatten(Layer):

    def __init__(self):
        self.input_matrix = 0
        pass

    def forward_propagation(self,input_matrix):
        self.input_matrix = input_matrix
        return input_matrix.flatten()

    def backward_propagation(self,output_error,learning_rate):

        return output_error.reshape(self.input_matrix.shape)