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


# # Second Iteration: Numpy Arrays converted to Torch Tensors 
# import torch

# dtype = torch.float
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# # N = Batch Size, D_in = input dimension, 
# # H = Hidden Dimension, D_out = output dimension
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Random input and output data
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)

# # Random Weights
# w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype)

# learning_rate = 1e-6
# for t in range(500):
#     # Forward Pass
#     h =  x.mm(w1)
#     h_relu = h.clamp(min=0)
#     y_pred = h_relu.mm(w2)

#     # Calculate and Print Loss to follow training progress
#     loss = (y_pred - y).pow(2).sum().item()
#     print(t, loss)

#     # Backpropogate loss to refine weights
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.t().mm(grad_y_pred)
#     grad_h_relu = grad_y_pred.mm(w2.t())
#     grad_h = grad_h_relu.clone()
#     grad_h[h < 0] = 0
#     grad_w1 = x.t().mm(grad_h)

#     # Update weights using gradient descent
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2
