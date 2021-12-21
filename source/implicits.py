import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numpy as np
import math
import geometry

"""
class _sphere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, r):
        x0 = x-c
        l = geometry.length(x0)

        ctx.save_for_backward(x0, l, r)

        return l - r

    @staticmethod
    def backward(ctx, df):
        x0, l, r = ctx.saved_tensors

        dx = x0 / l * df
        dc = -(x0 / l * df).view(-1, x0.shape[-1]).sum(0, keepdim=True)
        dr = -df.sum(0, keepdim=True).view(*r.shape)

        return dx, dc, dr

def sphere(x, c, r):
    return _sphere.apply(x, c, r)
"""

"""
Primitive SDFs
"""


class SphereSDF(nn.Module):
    def __init__(self, center, radius):
        super(SphereSDF, self).__init__()
        self.center = nn.parameter.Parameter(center)
        self.radius = nn.parameter.Parameter(radius)

    def forward(self, x):
        

        return geometry.length(x - self.center) - self.radius
    


class TriangleSDF(nn.Module):
    def __init__(self, v1, v2, v3):
        super(TriangleSDF, self).__init__()
        self.v1 = nn.parameter.Parameter(v1)
        self.v2 = nn.parameter.Parameter(v2)
        self.v3 = nn.parameter.Parameter(v3)

        self.compute_normal()

    def compute_normal(self):
        v1 = self.v1
        v2 = self.v2
        v3 = self.v3

        a1 = v2 - v1
        a2 = v3 - v2

        self.n = geometry.cross(a1, a2)

    def forward(self, x):
        # TODO : https://iquilezles.org/www/articles/triangledistance/triangledistance.htm

        return torch.sqrt(geometry.dot(x - self.v1, self.n) / geometry.dot(self.n, self.n))


class PlaneSDF(nn.Module):
    def __init__(self, v, n):
        super(PlaneSDF, self).__init__()

        self.v = nn.parameter.Parameter(v)
        self.n = nn.parameter.Parameter(n)

    def forward(self, x):
        return geometry.dot(x - self.v, self.n) / geometry.dot(self.n, self.n)


