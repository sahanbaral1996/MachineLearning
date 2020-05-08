import numpy as np
import math

class GaussianNaiveBayes:

    def __init__(self,data,label):
        self.classes = dict()
        self.data = np.c_[data,label]
        self.input_data = data
        self.label = self.data[:,-1]


    def separate_by_class(self):
        self.classes = {k:self.data[self.data[:,-1] == k] for k in np.unique(self.data[:,-1])}

    def mean(self,numbers):
        return sum(numbers)/float(len(numbers))

    def standard_deviation(self,numbers):
        return np.sqrt(np.sum((numbers-self.mean(numbers))**2)/(len(numbers)-1))

    def summarize(self,dataset):
        summary = [(self.mean(column),self.standard_deviation(column),len(column)) for column in zip(*dataset)]
        del(summary[-1])
        return summary

    def summary_by_class(self):
        summary = dict()
        self.separate_by_class()
        for keys in self.classes:
            summary[keys] = self.summarize(self.classes[keys])
        return summary

    def calculate_probability(self,x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * math.pi) * stdev)) * exponent

    def calculate(self,row):
        probability = dict()
        print(row)
        summary = self.summary_by_class()
        for key in self.classes:
            probability[key] = float(summary[key][0][2]/float(self.input_data.shape[0]))
            for index in range(4):
                probability[key] *= float(self.calculate_probability(row[index],summary[key][index][0],summary[key][index][1]))
        return probability