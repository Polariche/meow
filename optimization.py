import torch
import torch.nn as nn
import torch.nn.functional as F
import utils 
def gauss_newton(x, f):
    '''
    gauss-newton optimization
    '''

    y, dx = utils.y_dx(x, f)
    dx_pinv = torch.pinverse(dx.unsqueeze(-2))[..., 0]
 
    return x - y*dx_pinv
 
 
def lm(x, f, lamb = 1.1):
    '''
    levenberg-marquardt optimization
    for special cases where (dyi/dxj) is diagonal
    '''
    device = x.device

    y, dx = utils.y_dx(x, f)

    #compute in cpu?
    #x = x.to("cpu")
    #y = y.to("cpu")
    #dx = dx.to("cpu")

    J = dx.unsqueeze(-1)
    Jt = J.transpose(-2, -1)
    JtJ = torch.matmul(Jt, J)

    k = JtJ.shape[-1]
 
    diag_JtJ = torch.cat([JtJ[..., i, i] for i in range(k)])
    diag_JtJ = diag_JtJ.view(-1, k, 1)
    diag_JtJ = torch.eye(k, device=x.device).unsqueeze(0).expand(diag_JtJ.shape[0], -1, -1) * diag_JtJ
 
    pinv = torch.matmul(torch.inverse(JtJ + lamb * diag_JtJ), Jt)
    #pinv = torch.matmul(1 / (JtJ + lamb * diag_JtJ), Jt)
    delta = - pinv * y.unsqueeze(-1)
    delta = delta[..., 0, :]
 
    res = x + delta
    #res = res.to(device)

    return res
