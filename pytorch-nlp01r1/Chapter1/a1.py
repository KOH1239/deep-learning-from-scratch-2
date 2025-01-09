#!/usr/bin/python
# -*- coding: sjis -*-

import torch

def f(x):
    return (x[0]-2 * x[1]-1)**2 + (x[1] * x[2]-1)**2 + 1

def f_grad(x):
    z = f(x)
    z.backward()
    return x.grad

x = torch.tensor([1., 2., 3.], requires_grad=True)

for i in range(50):
    x = x - 0.1 * f_grad(x)
    x = x.detach().requires_grad_(True)
    print("x = ",x.data,", f = ",f(x).item())
    
