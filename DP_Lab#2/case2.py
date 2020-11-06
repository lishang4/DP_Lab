# -*- coding: UTF-8 -*-
'''
created at 2020/11/6
author: Lishang Chien
'''
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


class FuncAND:
    def __init__(self, dataset, weights, learningRate, epoch):
        self.dataset = np.array(dataset)
        self.w = np.array(weights)
        self.learningRate = learningRate
        self.epochs = epoch
        self.N = len(dataset)

    # calculate w0*x0 + w1*x1 ... wN*xN
    def n(self, w, x):
        return w.T.dot(x)

    # make sure n locate between 0 and 1
    def sigmoid(self, n):
        return 1.0 / (1.0 + np.exp(-n))

    # error function
    def cross_entropy(self, y, yHat):
        return y - yHat

    # logistic_regression algorithm
    def logistic_regression(self):
        errorMeasure_lastTime = 0.0 # initial last time's error measure
        for epoch in range(1,self.epochs):
            errorMeasure = 0.0 # reset error measure every epoch
            for x,y in dataset:
                yHat = self.sigmoid( self.n(self.w, x) ) # put n=sum(w*x) to o(n)=sigmoid(n), get yHat
                self.w = self.w.T + self.learningRate * self.cross_entropy(y, yHat) * np.array(x) # w = w + learning rate * y-yHat * x
                errorMeasure += self.cross_entropy(y, yHat) ** 2 # Mean Square Error(MSE), sum part
                self.learningRate *= 0.98  # 使learning rate逐漸下降

            errorMeasure /= self.N # Mean Square Error(MSE), 1/N part
            print(f'In Epoch {epoch}, bias = {self.w[:1][0]}, weights = {self.w[1:]}, error measure = {errorMeasure}')
            if errorMeasure < 0.05 or errorMeasure == errorMeasure_lastTime:
                return self.w, epoch # return result when error measure enough small or error measure stop beening smaller
            errorMeasure_lastTime = errorMeasure
                
        return self.w, self.epochs   # return result when error measure approach max epoch

    # classify x to positive and negative within y
    def classify_compoment(self, compoment):
        return compoment[compoment['y']==1], compoment[compoment['y']==0]
    
    # draw
    def fig_draw(self, weights):
        # clean-up data
        ps_train = [list(v[0])+[v[1]] for v in self.dataset] # mix tuple X and int Y into list

        # classify data by y's value
        ps_train = DataFrame(ps_train, columns=['x0', 'x1','x2', 'y'])
        train_positive, train_negative = self.classify_compoment(ps_train)

        # initial fig
        fig = plt.figure(num='Case 2, Func OR')
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")

        # draw points
        ax1.scatter(train_negative['x1'], train_negative['x2'], s=10, c='r', marker="x", label='y=0(train)')
        ax1.scatter(train_positive['x1'], train_positive['x2'], s=10, c='b', marker="o", label='y=1(train)')

        # draw line
        line_x = np.linspace(-1.5, 1.25)
        line_y = (-weights[0] - np.dot(weights[1], line_x)) / weights[2]
        ax1.plot(line_x, line_y, label='Decision Boundary')
        plt.legend(loc='upper left')
        print(f'>>[PROCESS] Fig drawn, close fig to end-up process')
        plt.show()

if __name__ == '__main__':
    # initial
    dataset = [((1, 0, 0), 0), #((x0,x1,x2), y1)
               ((1, 0, 1), 1),
               ((1, 1, 0), 1),
               ((1, 1, 1), 1)]
    weights = [0,0,0] # default weights
    learningRate = 1  # default laerning rate
    epoch = 1000      # maxiumn epoch

    # create case2
    case2 = FuncAND(dataset=dataset, weights=weights, learningRate=learningRate, epoch=epoch)

    # algorithm start
    print(f'>>[PROCESS] Func AND start, using logistic regression')
    w_trained,epoch = case2.logistic_regression()
    print(f'Final_weights : {w_trained}')
    print(f'Final_epoch : {epoch}')

    # draw
    case2.fig_draw(weights=w_trained)