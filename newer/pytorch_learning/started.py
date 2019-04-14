from __future__ import print_function
import torch
import numpy as np

x = torch.zeros(5, 3)

print(x)

y = torch.rand(5, 3)
print(y)

z = torch.zeros(5, 3, dtype=torch.long)
print(z)

a = torch.tensor([5, 3, 3])
print(a)

x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)
print(x.size())

y = torch.rand(5, 3)
print(x + y)

print(x)
print(y)
print(x.copy_(y))
print(x.t_())
print(x[:, 1])

#change the shape

x=torch.randn(4,4)
y = x.view(16)
print(x)
print(y)

#convert
a = torch.ones(5)
print(a)
b = a.numpy()

a=np.ones(5)
b=torch.from_numpy(b)
print(a,b)
print(torch.cuda.is_available())