import numpy as np
import pickle
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

    def forward_propagation(self, input_data):
        self.input_data = input_data
        for i in range(self.num_filters):
            for j in range(input_data.shape[2]):
                row = 0
                while row <= input_data.shape[0]-self.filter.shape[0]:
                    column = 0
                    while column <= input_data.shape[1]-self.filter.shape[1]:
                        ROI = input_data[row:row+3, column:column+3,j]
                        self.convoluted[:,:, i] = np.sum(ROI*self.filter[:,:,j,i])
                        column += 1
                    row +=1
        return self.convoluted

    def backward_propagation(self,output_error, learning_rate):
        in_error = np.zeros(self.input_size) #5,5,3
        w_error = np.zeros(self.filter.shape) # 3,3,3,8
        for i in range(self.num_filters):
            for j in range(self.input_data.shape[2]):
                row = 0
                while row <= self.input_data.shape[0] - output_error.shape[0]:
                    column = 0
                    while column <= self.input_data.shape[1] - output_error.shape[1]:
                        ROI = self.input_data[row:row + output_error.shape[0], column:column + output_error.shape[0], j]
                        w_error[:,:, j, i] = np.sum(ROI * output_error[:, :, i])
                        column += 1
                    row += 1
                row1 =0
                while row1 <= output_error.shape[0]-self.filter.shape[0]:
                    column1 =0
                    while column1 <= output_error.shape[1]-self.filter.shape[1]:
                        ROI1 = output_error[row1: row1+3,column1:column1+3,i]
                        in_error[:,:,j] = np.sum(ROI1 * self.filter[:,:,j,i])
                        column1 += 1
                    row1 += 1
        self.filter -= learning_rate * w_error
        return in_error



