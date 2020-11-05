import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

#網路上找的dataset 可以線性分割

dataset = np.array([
((1, 0, 0), 0),
((1, 1, 1), 0),
((1, 0, 1), 1),
((1, 1, 0), 1)])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(dataset, w):
    g = np.zeros(len(w))
    for x,y in dataset:
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        g += (error - y) * x
    return g / len(dataset)

def cost(dataset, w):
    total_cost = 0
    for x,y in dataset:
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        total_cost += abs(y - error)
    return total_cost

def logistic(dataset):
    w = np.zeros(3)
    epoch = 200
    eta = 1
    #costs = []
    for i in range(epoch):
        #current_cost = cost(dataset, w)
        print(f'In Epoch {i}, current_weights = {w}')
        #print( "current_cost=",current_cost)
        #costs.append(current_cost)
        w = w - eta * gradient(dataset, w)
        eta *= 0.95
    return w, i+1

# start
w,epoch = logistic(dataset)
print(f'Final_weights : {w}')
print(f'Final_epoch : {epoch}')

# draw
ps = [v[0] for v in dataset]
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter([v[1] for v in ps[:2]], [v[2] for v in ps[:2]], s=10, c='r', marker="x", label='X')
ax1.scatter([v[1] for v in ps[2:]], [v[2] for v in ps[2:]], s=10, c='b', marker="o", label='O')
l = np.linspace(-4,4)
a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(l, a*l + b, 'b-')
plt.legend(loc='upper left');
plt.show()