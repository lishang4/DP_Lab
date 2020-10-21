# -*- coding: UTF-8 -*-
'''
created at 2020/10/12

author: Lishang Chien
'''
from random import randint
from numpy import sign, linspace
from os.path import abspath
from constants import *
from pandas import DataFrame
from matplotlib.pyplot import pause, show, subplots, close, annotate

class Perceptron:

    # initial
    def __init__(self):
        self.threshold = THRESHOLD
        self.learningRate = LEARNINGRATE
        self.W = [0.3, 0.65, 1.0]
        self.rootPAth = abspath('.')
        self.fig,self.ax=subplots(figsize=(7, 7), num='Perceptron Graph')
        self.drawCompoment = []

    # loading data
    def load_file(self, fileName):
        data = []
        try:
            with open(f"{self.rootPAth}/data/{fileName}.txt", 'r') as txtFile:
                for f in txtFile:
                    f = f.replace('\n', '').split(',')
                    r = [int(_) for _ in f]
                    data.append(r)
            return data
        except Exception as e:
            raise

    def estimate_pla(self, W, X):
        return sign(self.inner_product(W, X))

    # calculate inner product within W & X
    def inner_product(self, W:list, X:list):
        return W[0]*X[0]+W[1]*X[1]-W[2]*self.threshold
        
    def set_up_canvas(self):
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.08, 1))
        self.ax.set_xlabel("X[1]")
        self.ax.set_ylabel("X[2]")
        self.ax.set_xlim(-28,28)
        self.ax.set_ylim(-28,28)

    # draw points
    def draw_train_points(self, x_compoment):
        x_positive = x_compoment[x_compoment['y']==1]
        x_negative = x_compoment[x_compoment['y']==-1]
        self.ax.scatter(x_positive['x1'],x_positive['x2'],marker="o",label="y=+1")
        self.ax.scatter(x_negative['x1'],x_negative['x2'],marker="x",label="y=-1")
        self.set_up_canvas()

    def draw_predict_points(self, x_compoment):
        x_positive = x_compoment[x_compoment['y']==1]
        x_negative = x_compoment[x_compoment['y']==-1]
        self.ax.scatter(x_compoment['x1'],x_compoment['x2'],marker="*",label="y=+1(test data)",color='r')
        self.ax.scatter(x_negative['x1'],x_negative['x2'],marker="s",label="y=-1(test data)", color='r')
        for x1, x2, y in zip(x_compoment['x1'], x_compoment['x2'], x_compoment['y']):
            annotate(
                f'({x1}, {x2}, {y})',
                xy=(x1, x2),
                ha='center',
                va='bottom',
                fontsize=8,
                color='r') #arrowprops=dict(arrowstyle='->', color='r', linewidth=1)
        self.set_up_canvas()
        pause(0.01)

    # draw line
    def draw_line(self):
        try:
            self.ax.lines.remove(lines[0])
        except Exception:
            pass
        x = linspace(-25,25, 50)
        y = (-self.W[2] - self.W[0] * x) / self.W[1]
        self.ax.plot(x, y)

    # reflesh canvas
    def reflesh_plt(self, X, _iter):
        self.ax.cla()
        self.draw_train_points(self.drawCompoment)
        self.draw_line()
        self.ax.set_title(f"            iter={_iter}\n \
            X1={X[0]}, X2={X[1]}, Y={X[2]}\n \
            W1={self.W[0]}, W2={self.W[1]}, W3={self.W[2]}\n \
            Learning Rate = {self.learningRate}", fontsize=8)
        pause(0.01)

     # train data flow using PLA
    def train(self):
        data = self.load_file('train')
        self.drawCompoment = DataFrame(data, columns=['x1', 'x2', 'y'])
        epoch = 0
        while epoch <= 100:
            misclassified = False
            for X in data:
                outputLabel = X[2]
                estimatedOutput_pre = self.estimate_pla(self.W, X)
                # in-case that np.sign return 0, make 0 to 1, two lines to avoid func call by multi times
                estimatedOutput = estimatedOutput_pre+1 if estimatedOutput_pre == 0 else estimatedOutput_pre
                if estimatedOutput != outputLabel:
                    misclassified = True
                    for i in range(0, len(X)-1):
                        # wk = wk + y * xk * learningRate
                        self.W[i] += outputLabel * X[i] * self.learningRate
                    print(f'updated weights to {self.W}')
                    self.reflesh_plt(X, epoch)
            if not misclassified:
                print(f'')
                print(f'train finished!') 
                print(f'final epoch: {epoch}')
                self.reflesh_plt(X, epoch)
                return
            epoch += 1
        print('epoch limited attach, loop stop')

    # predict data using well-trainned weights
    def predict(self):
        result = []
        predict_data = self.load_file('test')
        print(f'')
        print(f'ready to predict: {predict_data}')
        print(f'using well-trained weights: {self.W}')
        for i,data in enumerate(predict_data):
            data.append(self.estimate_pla(self.W, data))
            print(f'{data} is predicted to {data[2]}')
        # draw predict points to canvas
        self.draw_predict_points(DataFrame(predict_data, columns=['x1', 'x2', 'y']))
        show()
        close()


if __name__ == "__main__":
    model = Perceptron()
    model.train()
    model.predict()