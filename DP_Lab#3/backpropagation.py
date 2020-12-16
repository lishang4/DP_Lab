# -*- coding: UTF-8 -*-
'''
created at 2020/11/6
author: Lishang Chien
'''
import numpy as np
from os.path import abspath, join


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def cross_entropy(a, y):
    return a - y

class Loader:
    """define function for loading txt file"""

    @staticmethod
    def load_img(path):
        """load train_img/test_img txt file"""
        data = []
        try:
            with open(f"{abspath('.')}{path}", 'r') as txtFile:
                for line in txtFile:
                    line = line.replace('\n', '').split(',')
                    x = [int(_) for _ in line]
                    data.append(np.array(x))
            return data
        except Exception as e:
            raise

    @staticmethod
    def load_label(path):
        """load train_label/test_label txt file"""
        data = []
        try:
            with open(f"{abspath('.')}{path}", 'r') as txtFile:
                for line in txtFile:
                    x = line.replace('\n', '').split(',')
                    a = [0,0,0]
                    if x[0] == '0':
                        a = [1,0,0]
                    elif x[0] == '1':
                        a = [0,1,0]
                    elif x[0] == '2':
                        a = [0,0,1]
                    data.append(np.array(a))
            return data
        except Exception as e:
            raise

    @staticmethod
    def merge(imgFile, labelFile):
        return [(x, y) for x, y in zip(imgFile, labelFile)]

class NeuralNetwork:

    def __init__(self, layers, activation=sigmoid, activation_prime=sigmoid_prime, cost=cross_entropy):
        """ initial dependency / hyperparamters """
        # set activation function
        self.activation = activation
        self.activation_prime = activation_prime
        self.cost = cost

        # set-up
        self.weights = []
        self.learningRate = 0

        # layers = [784,30,3] means 
        # input layer has 784 neurons
        # one hidden layer has 30 neurons
        # output layer has 3 neurons
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learningRate=0.5, epochs=500):
        """ do the job """
        ones = np.atleast_2d(np.ones(X.shape[0])) # add column of ones to X
        X = np.concatenate((ones.T, X), axis=1) # add the bias unit to the input layer
        self.learningRate = learningRate
        errorMeasure_lastTime = 0
         
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]] # a = x[0]
            errorMeasure = 1

            # Feedforward part: 
            #   calculate a[l] to a[L], acivation function using sigmoid
            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)

            # Backward part: 
            #   output layer -> delta L
            error = self.cost(a[-1], y[i])
            deltas = [error]
            #   from second to last layer -> delta L-1,L-2,...,1
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            #   reverse deltas to simplify backpropagation's implementation(L -> l)
            deltas.reverse()

            # Update weights part:
            #   using stotistic backpropagation
            for l in range(len(self.weights)):
                layer = np.atleast_2d(a[l])
                delta = np.atleast_2d(deltas[l])
                self.weights[l] -= self.learningRate * np.dot(layer.T, delta)
            #errorMeasure = error ** 2 # Mean Square Error(MSE), sum part
            self.learningRate *= 0.98
            #errorMeasure /= len(X) # Mean Square Error(MSE), 1/N part
            #print(errorMeasure)
            #if errorMeasure < 0.02:
            #    return k # return result when error measure enough small or error measure stop beening smaller
            #errorMeasure_lastTime = errorMeasure

            if k % 100 ==0:print('epochs:', k)
        return epochs

    def predict(self, x): 
        """ predict unknown data(a) from well-trainned weights """
        a = np.concatenate((np.ones(1).T, np.array(x)))      
        for l in range(len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))

        return a

    def get_accuracy(self, X, Y):
        """ calculate accuracy rate and accuracy number """
        signal = []
        for x,y in zip(X, Y):
            result = list(nn.predict(x))
            predictNumber = result.index(max(result))
            answerNumber = list(y).index(max(y))
            signal.append(1 if answerNumber==predictNumber else 0)

        accuracyRate = round(signal.count(1)/(len(signal))*100,2)
        accuracyNum = f'{signal.count(1)} / {len(signal)}'
        return accuracyRate, accuracyNum

if __name__ == '__main__':

    # loading data
    fi1e = Loader()
    X = fi1e.load_img('/data/train_img.txt')
    Y = fi1e.load_label('/data/train_label.txt')
    unknown_X = fi1e.load_img('/data/test_img.txt')

    # normalization & grouping
    train_X = np.array(X[:6000])
    train_Y = np.array(Y[:6000])
    validation_X = np.array(X[-2000:])
    validation_Y = np.array(Y[-2000:])

    # neural network create
    nnStructure = [784,30,3]
    nn = NeuralNetwork(nnStructure)

    # trainning process
    print(f'>> Process starting ...')
    epochFinal = nn.fit(train_X,
                        train_Y, 
                        learningRate=0.45, 
                        epochs=1000)
    print(f'>> Trainning finished !')

    # valadation process
    print(f'>> Starting validation ...')
    accRateT, accNumT = nn.get_accuracy(train_X, train_Y)
    accRateV, accNumV = nn.get_accuracy(validation_X, validation_Y)
    print(f'>> Validation finished !')

    # predict process
    print(f'>> Starting predict ...')
    result = []
    for x in unknown_X:
        output = list(nn.predict(x))
        predict_number = output.index(max(output))
        result.append(predict_number)
    print(f'>> Predict finished !')
    #print(result)

    print(f'')
    print(f'###### Details ######')
    print(f'1.Data Count')
    print(f' -  Trainning data: {len(train_X)}')
    print(f' - Validation data: {len(validation_X)}')
    print(f' -    Predict data: {len(unknown_X)}')
    print(f' -           Total: {len(train_X)+len(validation_X)+len(unknown_X)}')
    print(f'')
    print(f'2.Hidden layer')
    print(f' -     Hidden Layer count: {len(nnStructure)-2}')
    for cnt in range(1,len(nnStructure)-1):
        print(f' - Layer#{cnt}\'s Neuron count: {nnStructure[cnt]}')
    print(f'')
    print(f'3.Final epoch: {epochFinal}')
    print(f'4.Final learning rate: {nn.learningRate}%')
    print(f'5.Accuracy Rate')
    print(f' -       Bias: {accRateT}% [{accNumT}]')
    print(f' - Validation: {accRateV}% [{accNumV}]')
    print(f'')
    print(f'6.Error Rate')
    print(f' -     Bias: {round(100-accRateT, 2)}%')
    print(f' - Variance: {round(-accRateV+accRateT, 2)}%')
    print(f'')
    print(f'>> Process End-Up !')