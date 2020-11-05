import matplotlib.pyplot as plt
import numpy as np
import math
import random
# pre load sklearn iris datasets
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
dataset = np.array([
((1, 0, 0), 0),
((1, 1, 1), 0),
((1, 0, 1), 1),
((1, 1, 0), 1)])

X = [v[0] for v in dataset]
print(X)
Y = [v[1] for v in dataset]
print(Y)

dataset = []

target_label = 0 # choose the target label of flower type
for index, x in enumerate(X):
    transform_label = None
    if Y[index] == target_label:
        transform_label = 1 # is the type
    else:
        transform_label = 0 # is not the type
    x = [x[0], x[2]]
    dataset.append((x,transform_label))
    
dataset = np.array(dataset)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sgd(dataset, w):
    #run sgd randomly
    index = random.randint(0, len(dataset) - 1)
    x, y = dataset[index]
    x = np.array(x)
    error = sigmoid(w.T.dot(x))
    g = (error - y) * x
    return g

def cost(dataset, w):
    total_cost = 0
    for x,y in dataset:
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        total_cost += abs(y - error)
    return total_cost

def logistic_regression(dataset):
    w = np.zeros(3)
    limit = 1500 #update times
    eta = 0.1 #update rate
    costs = []
    for i in range(limit):
        current_cost = cost(dataset, w)
        if i % 100 == 0:
            print ("epoch = " + str(i/100 + 1) + ": current_cost = ", current_cost)
        costs.append(current_cost)
        w = w - eta * sgd(dataset, w)
        eta = eta * 0.98 #decrease update rate
    plt.plot(range(limit), costs)
    plt.show()
    return w,(limit, costs)

def main():
    #execute
    w = logistic_regression(dataset)
    #draw 
    ps = [v[0] for v in dataset]
    label = [v[1] for v in dataset]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #plot via label
    tpx=[]
    for index, label_value in enumerate(label):
        px=ps[index][0]
        py=ps[index][1]
        tpx.append(px)
        if label_value == 1:
            ax1.scatter(px, py, c='b', marker="o", label='O')
        else:
            ax1.scatter(px, py, c='r', marker="x", label='X')

    l = np.linspace(min(tpx),max(tpx))
    a,b = (-w[0][0]/w[0][1], w[0][0])
    ax1.plot(l, a*l + b, 'g-')
    #plt.legend(loc='upper left');
    plt.show()

    limit = w[1][0]
    costs = w[1][1]
    w = w[0]

    # calculate score
    predicted_Y=[]
    answer_Y=[]
    for X,Y in dataset:
        answer_Y.append(Y)
        predicted_Y.append(sigmoid(w.T.dot(X)))
    predicted_Y = np.asarray(predicted_Y)
    predicted_Y = predicted_Y > 0.5
    #print(answer_Y)
    #print(predicted_Y)
    print ("Accuracy: ",str(accuracy_score(answer_Y, predicted_Y)*100)[:5],"%")

if __name__ == '__main__':
    main()