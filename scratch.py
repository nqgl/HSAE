import torch
import gradcool_functions

t = torch.ones(10)
t[3:7] *= 2
print(t)


v = torch.arange(10)
m = torch.max(torch.tensor(1), v - 5)
print(m)


x = torch.arange(7, dtype=torch.float) - 4.5
x = x/4
print(x)
x.requires_grad = True
y = gradcool_functions.undying_relu_extra_negative(x)
l = torch.sum(torch.pow(y + 2, 2))
l.backward()

print(x.grad)