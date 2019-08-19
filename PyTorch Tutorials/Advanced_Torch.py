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


# Iteration 5: Computational Graph Analysis via Contrast with TensorFlow
import tensorflow as tf
import numpy as np


# N = Batch Size, D_in = input dimension, 
# H = Hidden Dimension, D_out = output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Placeholders for the input and target data. Will be filled upon 
# Graph execution
x = tf.compat.v1.placeholder(tf.float32, shape=(None, D_in))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, D_out))

# Create TF Variables for weights and initialize randomly. TF Variables retain
# their value across computation graph executions.
w1 = tf.Variable(tf.random.normal((D_in, H)))
w2 = tf.Variable(tf.random.normal((H, D_out)))


# Forward Pass: Set up code for predicting y using TF Tensors.
# Does not actually perform numeric operations, just sets up computational graph
# for later execution.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# Compute Loss
loss = tf.reduce_sum((y- y_pred) ** 2.0)

# Compute loss gradients with respect to weights
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Update weights using gradient descent. To actually update weights, 
# new_w1 and new_w2 will have to be evaluated when the graph is executed. 
# In TF weight updating is a part of the computation graph, whereas in PyTorch
# this happens outside of the graph.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# With actual graph built, we enter a TensorFlow Session to execute the graph.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Create Numpy arrays to hold actual input and target data
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)

    for _ in range(500):
        loss_value, _, _ = sess.run([loss, new_w1, new_w2], 
                                    feed_dict={x: x_value, y: y_value})
        print(loss_value)