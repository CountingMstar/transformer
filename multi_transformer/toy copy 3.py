import torch
import torch.nn as nn

# Create an nn.Linear module
linear = nn.Linear(10, 5)

# Access the weight and bias attributes
weight = linear.weight
bias = linear.bias

# Print the shape of the weight and bias tensors
print("Weight shape:", weight.shape)
print("Bias shape:", bias.shape)

print(weight)
print(bias)