import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple
import numpy as np
def gauss_newton(x, f):
    '''
    gauss-newton optimization
    '''

    y = f(x)
    dx = torch.autograd.grad(y, [x], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    dx_pinv = torch.pinverse(dx.unsqueeze(-2))[..., 0]
 
    return x - y*dx_pinv
 
 
def lm(x, f, lamb = 1.1):
    '''
    levenberg-marquardt optimization
    '''
    device = x.device

    y = f(x)
    dx = torch.autograd.grad(y, [x], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]

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
 
    #pinv = torch.matmul(torch.pinverse(JtJ + lamb * diag_JtJ), Jt)
    pinv = torch.matmul(1 / (JtJ + lamb * diag_JtJ), Jt)
    delta = - pinv * y.unsqueeze(-1)
    delta = delta[..., 0, :]
 
    res = x + delta
    #res = res.to(device)

    return res

def _pers_cam(x, pose):
    cam_pos = x / x[..., -1].unsqueeze(-1)
    cam_pos = torch.matmul(cam_pos, pose.transpose(-1,-2))
    cam_pos = cam_pos[..., :2]
    

    return cam_pos

def _ortho_cam(x, pose):
    cam_pos = x
    cam_pos = torch.matmul(cam_pos, pose.transpose(-1,-2))
    cam_pos = cam_pos[..., :2]
    

    return cam_pos

class LMRayMarcher(nn.Module):
    '''
    Given position x, ray direction u and implicit-function f, 
    find the minimum distance d s.t. f(x+du) = 0
    '''
    def __init__(self, max_iter=20, lamb=1e-3):
        super(LMRayMarcher, self).__init__()
        self.max_iter = max_iter
        self.lamb = lamb 

    def forward(self, x, u, f):
        d = torch.zeros_like(x)[..., :1].requires_grad_(True)
        
        for i in range(self.max_iter):
            g = lambda d: f(x+d*u)
            d = lm(d, g, self.lamb)

        return d

class RayMarcher(nn.Module):
    def __init__(self, max_iter=20):
        super(RayMarcher, self).__init__()
        self.max_iter = max_iter

    def forward(self, x, u, f):
        d = torch.zeros_like(x)[..., :1].requires_grad_(True)
        
        for i in range(self.max_iter):
            d = d + f(x+d*u)

        return d

class OrthogonalCamera(nn.Module):
    def __init__(self, K):
        super(OrthogonalCamera, self).__init__()
        self.K = nn.parameter.Parameter(K)      # TODO : proper parametrization
    
    def forward(self, x):
        # x : (..., pose_batch, N, 3)

        return _ortho_cam(x, self.K)

        

class PerspectiveCamera(nn.Module):
    def __init__(self, K):
        super(PerspectiveCamera, self).__init__()
        self.K = nn.parameter.Parameter(K)
    
    def forward(self, x, pose):
        # x : (..., pose_batch, N, 3)

        return _pers_cam(x, self.K)

        


class PointToPixel(nn.Module):
    '''
    given a point cloud and its colors, render individual pixels in pixel space

    input : point cloud (..., n, 3), colors (..., n, k)
    output : img (..., k, h, w)
    '''
    def __init__(self, H:int, W:int, K: torch.Tensor, blendmode:str = 'average'):
        super(PointToPixel, self).__init__()
        self.H = H
        self.W = W
        self.K = K
        self.blendmode = blendmode

    def forward(self, x, c):
        if self.blendmode in ('min', 'max'):
            assert c.shape[-1] == 1, "blendmode 'min' or 'max' is only available for single channel" 

        original_shape = x.shape

        x = x.view(-1, *x.shape[-2:])
        c = c.view(-1, *c.shape[-2:])

        img = torch.zeros((x.shape[0], c.shape[-1], self.H, self.W), dtype=c.dtype, device=c.device)
        img_acc = torch.zeros((x.shape[0], 1, self.H, self.W), dtype=torch.long, device=c.device)

        if self.blendmode in ('min', 'max'):
            img_ind = -torch.ones((x.shape[0], 1, self.H, self.W), dtype=torch.long, device=c.device)

        Kx = _pers_cam(x, self.K)
        Kx = torch.round(Kx).long()


        for i, b_Kx in enumerate(Kx):
            cond_w = (b_Kx[:, 0] > 0) & (b_Kx[:, 0] < self.W)
            cond_h = (b_Kx[:, 1] > 0) & (b_Kx[:, 1] < self.H)
            cond_front = x[i, :, 2] > 0
            cond = cond_w & cond_h & cond_front

            b_Kx = b_Kx[cond]
            ind = b_Kx[:, 0] + b_Kx[:, 1] * self.H

            if self.blendmode is 'average':
                img[i] = img[i].view(c.shape[-1], self.H * self.W).index_add(1, ind, c[i, cond].T).view(c.shape[-1], self.H, self.W)
                img_acc[i] = img_acc[i].view(1, self.H * self.W).index_add(1, ind, torch.ones((1, ind.shape[0]), device=c.device, dtype=torch.long)).view(1, self.H, self.W)
            
            elif self.blendmode in ('min', 'max'):
                inf = np.inf if self.blendmode is 'min' else -np.inf
                cmp = min if self.blendmode is 'min' else max

                best_vals = [inf for _ in range(self.H * self.W)]
                best_inds = [-1 for _ in range(self.H * self.W)]

                for j, _ind in enumerate(ind):
                    val = c[i, cond][j].squeeze().item()

                    if cmp(best_vals[_ind], val) == val:
                        best_vals[_ind] = val
                        best_inds[_ind] = j

                pixindex = [i for i in range(self.H * self.W) if best_inds[i] != -1]
                
                pixindex = torch.tensor(pixindex, device=c.device)
                best_inds = torch.tensor(best_inds, dtype=torch.long, device=c.device)
                best_inds = best_inds[pixindex]

                ptvals = c[i, cond][best_inds].T

                img[i] = img[i].view(c.shape[-1], self.H * self.W).index_add(1, pixindex, ptvals).view(c.shape[-1], self.H, self.W)

        img_acc[img_acc == 0] = 1
        img = img / img_acc

        img = img.view(*original_shape[:-2], c.shape[-1], self.H, self.W)
        

        if self.blendmode in ('min', 'max'):
            return img
        
        else:
            return img



class PointFromPixel(nn.Module):
    '''
    given a point cloud and an image, sample colors corresponding to each point
    out-of-range points are recorded in "validity" as 0, others are recorded as 1

    input : point cloud (..., n, 3), img (..., k, h, w)
    output : point cloud (..., n, k), validity (..., n, 1)
    '''
    def __init__(self, K: Optional[torch.Tensor]=None):
        super(PointFromPixel, self).__init__()
        self.K = K

    def forward(self, x, img):
        H = img.shape[-2]
        W = img.shape[-1]
        C = img.shape[-3]

        if self.K is None:
            hH, hW = 0.5*H, 0.5*W

            K = torch.tensor(
                 [[hW, 0, hW],
                  [0, hH, hH],
                  [0, 0, 1]], 
                  dtype=torch.float, device=x.device)
        else:
            K = self.K

        original_shape = x.shape

        x = x.view(-1, *x.shape[-2:])
        img = img.view(-1, C, H, W)

        feat = torch.zeros((*x.shape[:-1], C), dtype=img.dtype, device=img.device)
        valid = torch.zeros((*x.shape[:-1], 1), dtype=torch.long, device=x.device)

        Kx = x / x[..., 2:3]
        Kx = torch.matmul(Kx, K.T)
        Kx = torch.round(Kx).long()


        for i, b_Kx in enumerate(Kx):
            cond_w = (b_Kx[:, 0] > 0) & (b_Kx[:, 0] < W)
            cond_h = (b_Kx[:, 1] > 0) & (b_Kx[:, 1] < H)
            cond_front = x[i, :, 2] > 0
            cond = cond_w & cond_h & cond_front

            b_Kx = b_Kx[cond]
            ind = b_Kx[:, 0] + b_Kx[:, 1] * H
            
            feat[i, cond] = img[i].view(C, H * W).index_select(1, ind).T
            valid[i] = cond.long().unsqueeze(-1)

        feat = feat.view(*original_shape[:-1], C)
        valid = valid.view(*original_shape[:-1], 1)

        return feat, valid
