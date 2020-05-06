import numpy as np
np.set_printoptions(suppress=True)


class LogisticRegression:

    @staticmethod
    def sigmoid(input_data):
        return 1/(1+np.exp(-input_data))

    @staticmethod
    def der_sigmoid(input_data):
        return LogisticRegression.sigmoid(input_data)*(1-LogisticRegression.sigmoid(input_data))

    def __init__(self):
        self.bias = np.random.rand(1)-0.5
        self.learning_rate = 0.01
        self.theta = None
        pass

    def fit(self,X,Y):
        self.theta = np.random.rand(X.shape[1])-0.5
        for epoch in range(30):
            costs = []
            error = 0
            for i in range(X.shape[0]):
                pred = np.dot(X[i],self.theta)+self.bias
                pred = self.sigmoid(pred)
                error += self.cost(pred,Y[i])
                self.gradient_descent(Y[i],pred,i,X)
            print(f"error for epoch {epoch} is {error/X.shape[0]}")

    def cost(self,predicted,actual):
        return (-actual*np.log(predicted))-((1-actual)*np.log(1-predicted))

    def gradient_descent(self,actual,prediction,index,X):
        self.bias -= (self.learning_rate * (prediction-actual))

        self.theta -= (self.learning_rate*(prediction-actual)*X[index])

    def predict(self,X):
        return self.sigmoid(np.dot(X,self.theta)+self.bias)