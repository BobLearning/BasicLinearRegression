import numpy as np

x = np.array([np.random.randint(10, size=10), np.random.randint(10, size=10), np.random.randint(10, size=10)])
w = [0, 0, 0]
b = 0

y = np.array([x[0] * 5, x[1] * 9, x[2] * 12]) + 5


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


print(gradient_descent(x, y, w, b, 10000, 0.001))