import torch

a = torch.ones(3, 2, 5)
# b = torch.rand(2, 5)
b = torch.ones(2, 5)

print(a.shape)
print(a)

print(b.shape)
print(b)

c = a + b
print(c.shape)
print(c)

b = b.expand(3, 2, 5)
print(b.shape)
print(b)

c = a + b
print(c.shape)
print(c)


"""
new_a = a[:, :, :3]
print(new_a)

new_b = b[:, 3:]
print(new_b.shape)
print(new_b)
new_b = new_b.expand(3, 2, 2)
print(new_b.shape)
print(new_b)

print('&&&&&&&&&&')
print(new_a.shape)
print(new_b.shape)
new = torch.cat([new_a, new_b], 2)
print(new)
"""

# x = torch.tensor([1, 2, 3])
# y = torch.tensor([4, 5, 6])
# # d = torch.cat([x, y], dim=1)

# random = torch.rand(2, 5)
# print(random)

# a[1] = random
# print(a)
# a[1, :] = 2
# print(a)

# # d = torch.cat([a, b], dim=1)
# # print(d.shape)
# # print(d)