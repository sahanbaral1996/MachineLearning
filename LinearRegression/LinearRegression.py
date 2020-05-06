import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


class LinearRegression:
    preds = []
    def __init__(self,input,output,lr):
        self.theta = np.random.rand(input.shape[1])-0.5
        self.input = input
        self.label = output
        self.bias = np.random.rand(1)-0.5
        self.learning_rate = lr

    def fit(self,epochs,preprocess= True):
		if preprocess:
			self.pre_process()
        for epoch in range(epochs):
            err = 0
            prediction = 0
            for sample in range(self.input.shape[0]):
                prediction = np.dot(self.input[sample],self.theta)+self.bias
                err += self.error(self.label[sample],prediction)
                self.gradient_descent(sample,self.label[sample],prediction,self.input[sample])
            print(f"error for epoch {epoch} is {err/self.input.shape[0]}")

    def gradient_descent(self, index,actual, predicted,input_data):
        self.bias += self.learning_rate*(actual-predicted)
        self.theta += self.learning_rate*(input_data*(actual-predicted))

    def error(self,actual,predicted):
        return ((actual-predicted)**2)/2

    def predict(self,X):
        return np.dot(X,self.theta)+self.bias

    def pre_process(self):
        for feature in range(self.input.shape[1]):
            self.input = (self.input-np.nanmin(self.input))/(np.nanmax(self.input)-np.nanmin(self.input))

