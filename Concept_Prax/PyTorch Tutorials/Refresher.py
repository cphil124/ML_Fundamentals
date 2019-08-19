# First Iteration: Only Numpy

import numpy as np

# N = Batch Size, D_in = input dimension, 
# H = Hidden Dimension, D_out = output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Random input and output data
x = np.random.randint(N, D_in)
y = np.random.randint(N, D_out)

# Random initial Weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    # forward pass
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Loss Calculation
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # BackPropogation. Compute new weights with respect to loss
    grad_y_pred = 2.0*(y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update Weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2