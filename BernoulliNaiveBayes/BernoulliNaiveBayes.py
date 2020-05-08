import numpy as np


class BNaiveBayes:

    def __init__(self, input_data):
        self.input_data = input_data
        self.feature = self.input_data[:,0:self.input_data.shape[1]-1]
        self.label = self.input_data[:,-1]
        self.unique_label = np.unique(self.label,axis=0)
        self.classes = dict()

    '''
    This method generated likelihood table on basis of features in a dictionary
    i.e
    fyi : 0,1,2 are feature index
        outlook = 0| humid = 1| wind = 2
            sunny       high        weak   (this function presumes sunny be 0, high be 0,weak be)
            (you can take any discrete number)
    
    output  0=>[[],[],[]],1=>[[],[]]....... and so on 
    ndarray inside a feature index is designed as :
        Yes    |    No
     a  3/10        2/4
     b
     c
    '''
    def likelihood(self):
        for i in range(self.feature.shape[1]):

            unique_vector = np.unique(self.feature[:, i], axis=0)
            frequency_table = []
            for j in unique_vector:
                frequency_row = []
                for k in self.unique_label:
                    count = np.count_nonzero(np.logical_and(self.label == k, self.feature[:, i] == j))
                    frequency_row.append(count/np.count_nonzero(self.label == k))
                frequency_table.append(frequency_row)
            self.classes[i] = frequency_table

    def predict(self, X):
        self.likelihood()
        probs = dict()
        likelihoods = []
        for i in self.unique_label:
            probX = 1
            for index,j in enumerate(X):
                probX *= (self.classes[index][j][i]*(np.count_nonzero(self.label == i)/self.label.shape[0]))/(np.count_nonzero(self.feature == j)/self.label.shape[0])
            likelihoods.append(probX*(np.count_nonzero(self.label == i)/self.label.shape[0]))

        probs = {index:likelihoods[index]/(np.sum(likelihoods)) for index,j in enumerate(likelihoods)}
        return probs