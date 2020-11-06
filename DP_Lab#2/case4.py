# -*- coding: UTF-8 -*-
'''
created at 2020/11/6
author: Lishang Chien
'''
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


class TrainExample:
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
            for x,y in self.dataset:
                yHat = self.sigmoid( self.n(self.w, x) ) # put n=sum(w*x) to o(n)=sigmoid(n), get yHat
                self.w = self.w.T + self.learningRate * self.cross_entropy(y, yHat) * np.array(x) # w = w + learning rate * y-yHat * x
                errorMeasure += self.cross_entropy(y, yHat) ** 2 # Mean Square Error(MSE), sum part
                self.learningRate *= 0.98  # 使learning rate逐漸下降

            errorMeasure /= self.N # Mean Square Error(MSE), 1/N part
            print(f'In Epoch {epoch}, bias = {self.w[:1][0]}, weights = {self.w[1:]}, error measure = {errorMeasure}')
            if errorMeasure < 0.2:
                return self.w, epoch # return result when error measure enough small or error measure stop beening smaller
            errorMeasure_lastTime = errorMeasure
                
        return self.w, self.epochs   # return result when error measure approach max epoch

    def predict(self, dataset, w):
        predicted_data = []
        for x in dataset:
            yHat = self.sigmoid( self.n(w, x) )
            result = 1 if yHat >= 0.5 else 0
            predicted_data.append(result)
            print(f'data{x} is predict to {result}')
        return predicted_data

    # classify x to positive and negative within y
    def classify_compoment(self, compoment):
        return compoment[compoment['y']==1], compoment[compoment['y']==0]

    # draw
    def fig_draw(self, weights, dataset_train, dataset_predict):
        # clean-up data
        ps_train = [list(v[0])+[v[1]] for v in dataset_train] # mix tuple X and int Y into list
        ps_predict=[list(v) for v in dataset_predict] # tuple value into list

        # classify data by y's value
        ps_train = DataFrame(ps_train, columns=['x0', 'x1','x2', 'y'])
        ps_predict = DataFrame(ps_predict, columns=['x0', 'x1','x2', 'y'])
        train_positive, train_negative = self.classify_compoment(ps_train)
        predi_positive, predi_negative = self.classify_compoment(ps_predict)

        # initial fig
        fig = plt.figure(num='Case 4, training example')
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel("height(x1")
        ax1.set_ylabel("weight(x2)")

        # draw points
        ax1.scatter(train_negative['x1'], train_negative['x2'], s=10, c='r', marker="x", label='y=0(train)')
        ax1.scatter(train_positive['x1'], train_positive['x2'], s=10, c='b', marker="o", label='y=1(train)')
        ax1.scatter(predi_negative['x1'], predi_negative['x2'], s=15, c='r', marker="s", label='y=0(predict)')
        ax1.scatter(predi_positive['x1'], predi_positive['x2'], s=15, c='b', marker="s", label='y=1(predict)')

        # draw line
        line_x = np.linspace(0,200)
        line_y = (-weights[0] - np.dot(weights[1], line_x)) / weights[2]
        ax1.plot(line_x, line_y, label='Decision Boundary')
        plt.legend(loc='upper left')
        print(f'>>[PROCESS] Fig drawn, close fig to end-up process')
        plt.show()

if __name__ == '__main__':
    # initial
    dataset = [((1, 120, 40), 0), #((x0,x1,x2), y1)
               ((1, 110, 35), 0),
               ((1,  90, 15), 0),
               ((1, 130, 30), 0),
               ((1, 170, 80), 1),
               ((1, 165, 55), 1),
               ((1, 150, 45), 1), 
               ((1, 180, 70), 1),
               ((1, 175, 65), 1),
               ((1, 160, 60), 1)]
    weights = [0,80,20] # default weights
    learningRate = 1  # default laerning rate
    epoch = 1000      # maxiumn epoch

    # create case1
    case4 = TrainExample(dataset=dataset, weights=weights, learningRate=learningRate, epoch=epoch)

    # algorithm start
    print(f'>>[PROCESS] TrainExample start, using logistic regression')
    w_trained,epoch = case4.logistic_regression()
    print(f'Final_weights : {w_trained}')
    print(f'Final_epoch : {epoch}')

    dataset_predict = [((1, 170, 60)), #((x0,x1,x2), y1)
                       ((1,  85, 15)),
                       ((1, 145, 45))]

    predicted_answer = case4.predict(dataset=dataset_predict, w=np.array(w_trained))
    predicted_data = [list(x)+[y] for x,y in zip(dataset_predict, predicted_answer)]

    # draw
    case4.fig_draw(weights=w_trained, dataset_train=dataset, dataset_predict=predicted_data)