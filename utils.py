import torch

def y_dx(x, f, retain_graph=True, create_graph=False):
    y = f(x)
    dx = torch.autograd.grad(y.sum(), [x], retain_graph=retain_graph, create_graph=create_graph)[0]

    return y, dx