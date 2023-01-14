import torch

x = torch.randn(3,6,13)
x_t = x[:, 0][:, None, :]
print(x.shape)
print(x_t.shape)
print(x)
print(x_t)
