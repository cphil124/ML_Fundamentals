# # Iteration 4: Custom Implementation of Autograd backward pass

# import torch

# class MyReLU(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, input):
#         """
#         Input is a Tensor containing the input. Returns a Tensor containing the 
#         output. ctx is a contextual object that can be used to stash information for
#         backward propogation.
#         """
#         ctx.save_for_backward(input)
#         return input.clamp(min=0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass, the Tensor containing the gradient of the loss with respect
#         to the output, and compute the gradient of the loss with respect to the input.
#         """
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         return grad_input

    
# dtype = torch.float
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# # N = Batch Size, D_in = input dimension, 
# # H = Hidden Dimension, D_out = output dimension
# N, D_in, H, D_out = 64, 1000, 100, 10

# # # Random input and output data
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)

# # Random Weights
# # Need to set requires_grad=True in order to be able to calculate the gradient and
# # Use the built-in backpropogation functionality. 
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# learning_rate = 1e-6
# for t in range(500):
#     relu = MyReLU.apply

#     # Forward Pass: Get prediction using custom ReLU autograd function
#     y_pred = relu(x.mm(w1)).mm(w2)

#     # Compute and print loss
#     loss = (y_pred - y).pow(2).sum()
#     print(t, loss.item())

#     # Use custom autograd to perform backward pass
#     loss.backward()

#     # Update Weights with gradient descent
#     with torch.no_grad():
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad

#         # Manually zero the gradients after updating weights
#         w1.grad.zero_()
#         w2.grad.zero_()


# # Iteration 5: Computational Graph Analysis via Contrast with TensorFlow
# import tensorflow as tf
# import numpy as np


# # N = Batch Size, D_in = input dimension, 
# # H = Hidden Dimension, D_out = output dimension
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Placeholders for the input and target data. Will be filled upon 
# # Graph execution
# x = tf.compat.v1.placeholder(tf.float32, shape=(None, D_in))
# y = tf.compat.v1.placeholder(tf.float32, shape=(None, D_out))

# # Create TF Variables for weights and initialize randomly. TF Variables retain
# # their value across computation graph executions.
# w1 = tf.Variable(tf.random.normal((D_in, H)))
# w2 = tf.Variable(tf.random.normal((H, D_out)))


# # Forward Pass: Set up code for predicting y using TF Tensors.
# # Does not actually perform numeric operations, just sets up computational graph
# # for later execution.
# h = tf.matmul(x, w1)
# h_relu = tf.maximum(h, tf.zeros(1))
# y_pred = tf.matmul(h_relu, w2)

# # Compute Loss
# loss = tf.reduce_sum((y- y_pred) ** 2.0)

# # Compute loss gradients with respect to weights
# grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# # Update weights using gradient descent. To actually update weights, 
# # new_w1 and new_w2 will have to be evaluated when the graph is executed. 
# # In TF weight updating is a part of the computation graph, whereas in PyTorch
# # this happens outside of the graph.
# learning_rate = 1e-6
# new_w1 = w1.assign(w1 - learning_rate * grad_w1)
# new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# # With actual graph built, we enter a TensorFlow Session to execute the graph.
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     # Create Numpy arrays to hold actual input and target data
#     x_value = np.random.randn(N, D_in)
#     y_value = np.random.randn(N, D_out)

#     for _ in range(500):
#         loss_value, _, _ = sess.run([loss, new_w1, new_w2], 
#                                     feed_dict={x: x_value, y: y_value})
#         print(loss_value)


# # Iteration 6: Leveraging PyTorch NN Package
# import torch
# from torch import nn

# # N = Batch Size, D_in = input dimension, 
# # H = Hidden Dimension, D_out = output dimension
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Create random tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # Use nn package to define model as a sequence of layers. The nn.Sequential Module
# # contains other Modules and applies them in sequence to produce an output. 
# # The Linear Module computes output from input using a Linear function and holds internal 
# # Tensors for its weights and bias.
# model = nn.Sequential(
#     nn.Linear(D_in, H),
#     nn.ReLU(),
#     nn.Linear(H, D_out),
# )

# # NN module also contains various loss functions. For Example Mean Squared Error(MSE)
# loss_fn = nn.MSELoss(reduction='sum')

# learning_rate = 1e-4
# for t in range(500):
#     # Forward Pass: compute y by passing x to the model. Module objects override
#     # the __call__ operator so they can be called like functions. When doing so, 
#     # Simply pass a Tensor with the input and receive a Tensor with the output.
#     y_pred = model(x)

#     # Compute and print loss. The loss function takes predictions and labels as input
#     # and returns a tensor containing the loss.
#     loss = loss_fn(y_pred, y)
#     print(t, loss)

#     # Zero the gradient before running backward pass.
#     model.zero_grad()

#     # Backward Pass: Gradient of loss is computed with respect to all learnable parameters
#     # Internally parameters of each Module are stored internally in Tensors 
#     # with requires_grad=True, thus loss.backward will target all learnable parameters in the model
#     loss.backward()

#     with torch.no_grad():
#         for param in model.parameters():
#             param -= learning_rate * param.grad


# # Iteration 7: Leveraging PyTorch Optim Package
# import torch
# from torch import nn

# # N = Batch Size, D_in = input dimension, 
# # H = Hidden Dimension, D_out = output dimension
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Initialize random inputs and target outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # Define model and loss function with nn
# model = nn.Sequential(
#     nn.Linear(D_in, H),
#     nn.ReLU(),
#     nn.Linear(H, D_out)
# )
# loss_fn = nn.MSELoss(reduction='sum')

# # Define Optimizer using optim package that will update weights of model automatically
# # We will try the Adam Optimizer. 
# learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# for t in range(500):
#     # Forward Pass:
#     y_pred = model(x)

#     # Loss computation
#     loss = loss_fn(y_pred, y)
#     print(t, loss.item())

#     # Now using optimizer, reset gradient. Necessary as gradients accumulate across iterations
#     optimizer.zero_grad()

#     # Backward Pass: Gradient of loss with respect to parameters is calculated.
#     loss.backward()

#     # The Optimizer .step() function makes an update to the parameteres.
#     optimizer.step()


# # Iteration 8: Custom NN Module

# import torch

# class TwoLayerNet(torch.nn.Module):
#     def __init__(self, D_in, H, D_out):
#         """
#         Two Linear Modules are instantiated and assigned as member variables
#         """
#         super(TwoLayerNet, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)

    
#     def forward(self, x):
#         """
#         Accepts an input Tensor X and returns an output Tensor. Modules defined in the
#         constructor as well as arbitrary operators can be used on Tensors.
#         """
#         h_relu = self.linear1(x).clamp(min=0)
#         y_pred = self.linear2(h_relu)
#         return y_pred


# # N = Batch Size, D_in = Input dimension, 
# # H = Hidden Dimension, D_out = Output dimension
# N, D_in, H, D_out = 64, 1000, 100, 10

# # Initialize random inputs and target outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# # Instantiate model via class constructor
# model = TwoLayerNet(D_in, H, D_out)

# criterion = torch.nn.MSELoss(reduction='sum') # Formerly referred to as loss_fn
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
# for t in range(500):
#     # Forward Pass
#     y_pred = model(x)

#     loss = criterion(y_pred, y)
#     print(t, loss.item())

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()



# Iteration 9: Control Flow + Weight Sharing

import torch
import random

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        3 Linear instances are constructed for the forward pass
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_layer = torch.nn.Linear(H, H)
        self.output_layer = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For a dynamic forward pass, we'll randomly iterate over the middle layer
        between 0 and 3 times, each time reusing the same weights. 

        This being a dynamic graph, normal Python Flow control can be leveraged.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_layer(h_relu).clamp(min=0)
        y_pred = self.output_layer(h_relu)
        return y_pred

# N = Batch Size, D_in = Input dimension, 
# H = Hidden Dimension, D_out = Output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Initialize random inputs and target outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct Model
model = DynamicNet(D_in, D_out)

# Construct loss function and optimizer. Training this type of model is not 
# as effective with Stochastic Gradient Descent, so wwe use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.95)
for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()