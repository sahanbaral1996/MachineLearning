import numpy as np

class MaxPooling:

    def __init__(self, pool_size):
        self.input_data = 0
        self.pool_size = pool_size
        self.pooled = 0
        self.mask = 0

    def forward_propagation(self, input_data):
        self.input_data= input_data
        convoluted_x = input_data.shape[0] - self.pool_size[0] + 1
        convoluted_y = input_data.shape[1] - self.pool_size[1] + 1
        pooled = np.zeros((convoluted_x, convoluted_y))
        self.mask = np.zeros(input_data.shape)
        for i in range(1):
            row = 0
            while row <= input_data.shape[0] - self.pool_size[0]:
                column = 0
                while column <= input_data.shape[1] - self.pool_size[1]:
                    ROI = input_data[row:row + self.pool_size[0], column:column + self.pool_size[1]]
                    pooled[row, column] = np.max(ROI)
                    #self.mask = (ROI == np.max(ROI)).astype(int)
                    for i in range(ROI.shape[0]):
                        for j in range(ROI.shape[1]):
                            if ROI[i,j] == np.max(ROI):
                                self.mask[i+row,j+column] =1
                    column += 1
                row += 1
        self.pooled = pooled
        print(self.mask)
        return pooled

    def back_propagation(self, output_error):
        print(self.mask* self.input_data)
        print(self.pooled+output_error)
        in_error = self.input_data



a = np.array([[2,4,8,3,6],[9,3,4,2,5],[5,4,6,3,1],[2,3,1,3,4],[2,7,4,5,7]])
p = MaxPooling((2,2))
b = np.array([[1,3,1],[1,4,2],[6,2,1]])
p.forward_propagation(a)
print(p.back_propagation(b))