import numpy as np
import matplotlib.pyplot as plt

n = 20
size = 10

x = np.array([np.random.randint(n, size=size), np.random.randint(n, size=size), np.random.randint(n, size=size)])

w = [0, 0, 0]
b = 0

y = np.array([x[0] * 5, x[1] * 9, x[2] * 12]) + 5


def plot(w,b):
    y_predict = x * w[0] + b
    y_predict2 = x * w[1] + b
    y_predict3 = x * w[2] + b

    plt.plot(x, y, 'x')
    plt.plot(x, y_predict)
    plt.plot(x, y_predict2)
    plt.plot(x, y_predict3)

    plt.ylabel('Hours') 
    plt.xlabel('Pass or Fail') 

    plt.show()

def cost_function(x, y, w, b):
    m = x.shape[0]
    
    cost = 0
    for i in range(m):
        y_predict = x[i] * w[i] + b

        cost_i = (y_predict - y[i]) ** 2
        cost = cost + cost_i

    cost_function = np.sum((1/2*m) * cost)
        

    return cost_function

#print(cost_function(x, y, w, b))

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    W_Der = []

    for i in range(m):
        y_predict = x[i] * w[i] + b

        W_Der_i = W_Der.append(np.sum(np.dot(x, (y_predict - y[i])) / m))
        B_Der = np.sum(y_predict - y[i]) / m

    return W_Der, B_Der

#print(compute_gradient(x, y, w, b))

def gradient_descent(x, y, w, b, iterations, alpha):
    m = x.shape[0]

    for i in range(iterations):
        W_Der, B_Der = compute_gradient(x, y, w, b)

        for j in range(m):
            w[j] = w[j] - alpha * W_Der[j]
            b = b - alpha * B_Der

        if i%1000 == 0:
            print(f"Iteration: {i}, W = {w}, B = {b}")

    return w, b


W,B = gradient_descent(x, y, w, b, 100000, 0.00001)

plot(W,B)