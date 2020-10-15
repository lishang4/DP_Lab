# -*- coding: UTF-8 -*-
'''
created at 2020/10/12

author: Lishang Chien
'''
from random import randint
from numpy import sign
import os

class TrainningModel:

    # initial
    def __init__(self):
        self.data = []
        self.threshold= -1
        self.learningRate = 0.001
        self.W = [0.5, 0.5, 1.0]

    # loading data
    def load_file(self, fileName):
        data = []
        os.path.abspath('.')
        with open(f"C:/WorkSpace/hw_PLA_507170627/data/{fileName}.txt", 'r') as f:
            for ff in f:
                ff = ff.replace('\n', '').split(',')
                r = [int(d) for d in ff]
                data.append(r) 
        return data

    def inner_product(self, W:list, X:list):
        return W[0]*X[0]+W[1]*X[1]-W[2]*self.threshold

    def random_data(self):
        for i in range(0,200):
            L = []
            L.append(randint(0,100))
            L.append(randint(0,100))
            L.append(sign(L[0]+L[1]-120))
            self.data.append(L)

    def train_start(self):
        self.data = self.load_file('train')
        #self.random_data()
        epoch = 0
        while epoch <= 100:
            wrong = False
            for X in self.data:
                x0, x1 = X[0], X[1]
                outputLabel = X[2]
                estimatedOutput = sign(self.inner_product(self.W, X))
                if estimatedOutput != outputLabel and estimatedOutput != 0.0:
                    wrong = True
                    self.W[0] += (outputLabel-estimatedOutput) * x0 * self.learningRate #w1 = y-yHat*x1*learning rate(control range)
                    self.W[1] += (outputLabel-estimatedOutput) * x1 * self.learningRate #w2 = y-yHat*x2*learning rate(control range)
                    break
                print(self.W)
            if not wrong:
                print(f'trainning done') 
                return
            epoch += 1
        print('epoch apporch limited')

    def predict(self):
        predict_data = self.load_file('test')
        for data in predict_data:
            print(f'{data} is predicted for {sign(self.inner_product(self.W, data))}')

class PredictionModel:
    def __init_(self):
        pass

if __name__ == "__main__":
    model = TrainningModel()
    model.train_start()
    model.predict()