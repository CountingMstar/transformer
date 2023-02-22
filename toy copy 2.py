import torch.nn as nn
import torch

# create a 4D tensor with shape (batch_size, channels, height, width)
x = torch.rand(3, 2, 5)
print(x.shape)
print(x)

# x = x.view(3, -1)
# print(x.shape)
# print(x)
x = x.view(3*2, -1)
print(x.shape)
print(x)

x = x.view(3, 2, 5)
print(x.shape)
print(x)

# # reshape the tensor into a 2D tensor with shape (batch_size, channels * height * width)
# x = x.view(32, -1)

# # create a linear layer with input size = channels * height * width and output size = 10
# linear_layer = nn.Linear(x.size(1), 10)

# # apply the linear layer to the input tensor
# output = linear_layer(x)