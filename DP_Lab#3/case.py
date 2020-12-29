# -*- coding: UTF-8 -*-
'''
created at 2020/12/18
author: Lishang Chien
'''
import numpy as np
import random
from scipy.special import expit
import loader

#### Miscellaneous functions
def sigmoid(z, prime=False):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z)) if not prime else sigmoid(z)*(1.0-sigmoid(z))

def softmax(z , prime=False):
    # Numerically stable with large exponentials
    exps = np.exp(z - z.max())
    if prime: return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


class Network:

    def __init__(self, sizes: list):
        """ 
        sizes 包含各層神經元的數量, 假設建構一個2層的NN, 
        則目標會含有:input layer x1, hidden layer x1, output layer x1
        若每層的神經元數量分別是784,15,1則創建時導入list object [784,15,1]即可。

        example: net = Network([784, 15, 1]) 
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.train_acc = 0
        self.test_acc = 0
        self.epoch = 0
        self.eta = 0

    def feedforward(self, a, predict=False):
        """Return the output of the network if ``a`` is input."""
        zs = []
        activations = [a]

        activation = a
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        activations.append(softmax(zs[-1]))
        activations.pop(-2)

        return activations if predict else (zs, activations)

    def update_mini_batch(self, mini_batch, eta):
        """
        依照mini batch size 局部更新 weights & bias
        對每一個局部size的batch part進行backpropagation後更新超級參數

        mini_batch: tuple
        eta: int
        """
        batch_size = len(mini_batch)

        # transform to (input x batch_size) matrix
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        # transform to (output x batch_size) matrix
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        nabla_b, nabla_w = self.backprop(x, y)
        self.weights = [w - (eta / batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

        return

    def backprop(self, x, y):
        """ backpropagation part """
        nabla_b = [0 for i in self.biases]
        nabla_w = [0 for i in self.weights]

        # feedforward
        zs, activations = self.feedforward(x)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * softmax(zs[-1], prime=True)
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1]) # reshape to (n x 1) matrix
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid(z, prime=True)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta.sum(1).reshape([len(delta), 1]) # reshape to (n x 1) matrix
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def gradient_descent(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, eva_train_result=[], eva_test_result=[]):
        """
        每一個新的epoch皆先打亂training_data確保mini batch不會每次都用一樣的順序訓練
        此舉可避免梯度消失, learning rate每一批完整batch後以0.98比率漸小縮短擺盪程度
        停止條件：
            1.達到最大世代數
            2.訓練資料集準確率上升，而未參與訓練的驗證集準確率卻下降，避免Overfitting
        """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            eta *= 0.98
            self.epoch = epoch
            self.eta = eta
            if test_data:
                eva_train_result.append(self.evaluate_train(training_data))
                eva_test_result.append(self.evaluate_validation(test_data))
                print(f"Epoch {epoch}:bias {eva_train_result[-1]} / {n} ({eva_train_result[-1]/n*100}%)")
                print(f"Epoch {epoch}:validation {eva_test_result[-1]} / {n_test} ({eva_test_result[-1]/n_test*100}%)")
                try:
                    overfitting = eva_train_result[-2]/n > eva_train_result[-1]/n and \
                        eva_test_result[-2]/n_test < eva_test_result[-1]/n_test
                except:
                    overfitting = False
                if overfitting and epoch > 5:
                    print(f'bias raise up but validation down, stop training to avoid overfitting.')
                    self.train_acc = eva_train_result[-2]/n*100
                    self.test_acc = eva_test_result[-2]/n_test*100
                    break
                self.train_acc = eva_train_result[-1]/n*100
                self.test_acc = eva_test_result[-1]/n_test*100
            else:
                print(f"Epoch {epoch} complete")

    def evaluate_validation(self, test_data):
        """ 驗證:返回與output label相比正確的驗證資料數量 """
        eva_results = [(np.argmax(self.feedforward(x)[1][-1]), y)
                        for (x, y) in test_data]
        return sum(int(yHat == y) for (yHat, y) in eva_results)

    def evaluate_train(self, train_data):
        """ 驗證:返回與output label相比正確的訓練資料數量 """
        eva_results = [(np.argmax(self.feedforward(x)[1][-1]), np.argmax(y))
                        for (x, y) in train_data]
        return sum(int(yHat == y) for (yHat, y) in eva_results)

    def cost_derivative(self, yHat, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return yHat-y

    def predict(self, test_data):
        """ 預測輸入資料(input) """
        eva_results = [np.argmax(self.feedforward(x)[1][-1])
                        for x in test_data]
        return eva_results



if __name__ == '__main__':
    print(f'>> Process Start !')
    training_data, validation_data, test_data = loader.load_data_wrapper()

    print(f'creating neural netwotk ...')
    nn_structure = [784, 128, 64, 3]
    max_epoch = 15
    mini_batch_size = 32
    lr = 1
    net = Network(nn_structure)

    print(f'start training & evaluating ...')
    print(f'usage: max_epoch: {max_epoch}')
    print(f'       mode = mini batch')
    print(f'       mini-batch size = {mini_batch_size}')
    print(f'       init learning rate = {lr}')
    net.gradient_descent(training_data, \
        max_epoch, max_epoch, lr, test_data=validation_data)
    print(f'training finish !')

    print(f'now starting predict test img ...')
    loader.output('/data/test_label.txt', net.predict(test_data))
    print(f'predict succeed ! output file locate at: {"./test.txt"}')

    print(f'')
    print(f'****** Details ******')
    print(f'1.Data Count')
    print(f' -  Trainning data: {len(training_data)}')
    print(f' - Validation data: {len(validation_data)}')
    print(f' -    Predict data: {len(test_data)}')
    print(f' -           Total: {len(training_data)+len(validation_data)+len(test_data)}')
    print(f'')
    print(f'2.Hidden layer')
    print(f' -     Hidden Layer count: {len(nn_structure)-2}')
    for cnt in range(1,len(nn_structure)-1):
        print(f' - Layer#{cnt}\'s Neuron count: {nn_structure[cnt]}')
    print(f'')
    print(f'3.Final epoch: {net.epoch}')
    print(f'4.Final learning rate: {net.eta}%')
    print(f'5.Accuracy Rate')
    print(f' -       Bias: {net.train_acc}%')
    print(f' - Validation: {net.test_acc}%')
    print(f'')
    print(f'6.Error Rate')
    print(f' -     Bias: {round(100-net.train_acc, 2)}%')
    print(f' - Variance: {round(-net.test_acc+net.train_acc, 2)}%')
    print(f'')
    print(f'>> Process End-Up !')