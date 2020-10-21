# -*- coding: UTF-8 -*-
'''
created at 2020/10/12
author: Lishang Chien
'''
from os.path import abspath
from constants import *
from pandas import DataFrame
from numpy import (
    sign as np_sign, 
    linspace as np_linspace)
from matplotlib.pyplot import (
    pause as plt_pause,show as plt_show,
    subplots as plt_subplots,
    close as plt_close, 
    annotate as plt_annotate, 
    savefig as plt_savefig)

class Perceptron:

    # initial
    def __init__(self):
        self.threshold = THRESHOLD
        self.learningRate = LEARNINGRATE
        self.W = WEIGHTS
        self.epoch = EPOCH
        self.rootPAth = abspath('.')
        self.fig,self.ax=plt_subplots(figsize=(7, 7), num='Perceptron Graph Canvas(fig)')
        self.drawCompoment = []

    # load file
    def load_file(self, fileName):
        data = []
        try:
            with open(f"{self.rootPAth}/data/{fileName}.txt", 'r') as txtFile:
                for line in txtFile:
                    line = line.replace('\n', '').split(',')
                    data.append([int(_) for _ in line])
            return data
        except Exception as e:
            raise
    
    # get y_hat
    def estimate_pla(self, W, X):
        return np_sign(self.inner_product(W, X))

    # calculate inner product within W & X
    def inner_product(self, W:list, X:list):
        return W[0]*X[0]+W[1]*X[1]-W[2]*self.threshold

    # classify x to positive and negative within y
    def classify_compoment(self, compoment):
        return compoment[compoment['y']==1], compoment[compoment['y']==-1]

    # default setting for canvas(fig)    
    def set_up_canvas(self):
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.08, 1))
        self.ax.set_xlabel("X[1]")
        self.ax.set_ylabel("X[2]")
        self.ax.set_xlim(-28,28)
        self.ax.set_ylim(-28,28)

    # draw train data
    def draw_train_points(self, x_compoment):
        x_positive, x_negative = self.classify_compoment(x_compoment)
        self.ax.scatter(x_positive['x1'],x_positive['x2'],marker="o",label="y=+1")
        self.ax.scatter(x_negative['x1'],x_negative['x2'],marker="x",label="y=-1")
        self.set_up_canvas()

    # draw test data
    def draw_predict_points(self, x_compoment):
        x_positive, x_negative = self.classify_compoment(x_compoment)
        self.ax.scatter(x_positive['x1'],x_positive['x2'],marker="*",label="y=+1(test data)",color='r')
        self.ax.scatter(x_negative['x1'],x_negative['x2'],marker="s",label="y=-1(test data)", color='r')
        for x1, x2, y in zip(x_compoment['x1'], x_compoment['x2'], x_compoment['y']):
            plt_annotate(
                f'({x1}, {x2}, {y})',
                xy=(x1, x2+5),
                ha='center',
                va='bottom',
                fontsize=8,
                color='r') #if need arrow add this: arrowprops=dict(arrowstyle='->', color='r', linewidth=1)
        self.set_up_canvas()
        plt_pause(0.01) # use pause to mock-up event-loop which can make it actually draw the fig a least once

    # draw line
    def draw_line(self):
        # to simulation dynamic, remove last line before drawing new one
        try:
            self.ax.lines.remove(lines[0])
        except Exception:
            pass
        x = np_linspace(-25,25, 50)
        y = (-self.W[2] - self.W[0] * x) / self.W[1]
        self.ax.plot(x, y)

    # reflesh canvas(fig)
    def reflesh_canvas(self, X, epoch):
        self.ax.cla()
        self.draw_train_points(self.drawCompoment)
        self.draw_line()
        self.ax.set_title(f"            epoch={epoch}\n \
            X1={X[0]}, X2={X[1]}, Y={X[2]}\n \
            W1={self.W[0]}, W2={self.W[1]}, W3={self.W[2]}\n \
            Learning Rate = {self.learningRate}", fontsize=8)
        plt_pause(0.01) # use pause to mock-up event-loop which can make it draw the fig(a least once)

     # train data flow using PLA
    def train(self):
        data = self.load_file('train')
        self.drawCompoment = DataFrame(data, columns=['x1', 'x2', 'y'])
        for epoch in range(1, self.epoch):
            classified = True
            for X in data:
                outputLabel = X[2]
                estimatedOutput_pre = self.estimate_pla(self.W, X)
                estimatedOutput = estimatedOutput_pre if estimatedOutput_pre != 0 else estimatedOutput_pre+1
                if estimatedOutput != outputLabel:
                    classified = False
                    for i in range(0, len(X)-1):
                        self.W[i] += outputLabel * X[i] * self.learningRate # wk = wk + y * xk * learningRate
                    print(f'In Epoch {epoch}, weights got updated to {self.W}')
                    self.reflesh_canvas(X, epoch)
            if classified:
                print(f'')
                print(f'train finished!') 
                print(f'final epoch: {epoch}')
                print(f'final weights: {self.W}')
                self.reflesh_canvas(X, epoch)
                return
        print('epoch attached limit, stop estimating')

    # predict data using well-trainned weights
    def predict(self):
        predict_data = self.load_file('test')
        print(f'==========================')
        print(f'ready to predict: {predict_data}')
        print(f'using well-trained weights: {self.W}')

        # do predict using well-trained weights
        for i,data in enumerate(predict_data):
            data.append(self.estimate_pla(self.W, data))
            print(f'{data} is predicted to {data[2]}')
        print(f'predict finished!')
        print(f'')

        # draw predict points to canvas
        self.draw_predict_points(DataFrame(predict_data, columns=['x1', 'x2', 'y']))

        # save canvas(fig) to root path
        pngPath = f"{self.rootPAth}/perceptron_507170627.png"
        plt_savefig(pngPath)
        print(f'graph was saved to {pngPath}')

        # not plt_pause because there no event-loop anymore
        print(f'>> close canvas(fig) to end-up process')
        plt_show(block=True)
        plt_close()


if __name__ == "__main__":
    model = Perceptron()
    print(f'>> process up')
    model.train()
    model.predict()
    print(f'>> process down')