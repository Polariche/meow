import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple
import numpy as np
import geometry

import utils.optimization


class LMRayMarcher(nn.Module):
    '''
    Given position x, ray direction u and implicit-function f, 
    find the minimum distance d s.t. f(x+du) = 0
    '''
    def __init__(self, sdf:nn.Module, max_iter:int=20, lamb:float=1e-3):
        super(LMRayMarcher, self).__init__()
        self.max_iter = max_iter
        self.lamb = lamb 
        self.sdf = sdf

    def forward(self, x, u):
        d = torch.zeros_like(x)[..., :1].requires_grad_(True)
        
        for i in range(self.max_iter):
            g = lambda d: self.sdf(x+d*u)
            d = utils.optimization.lm(d, g, self.lamb)

        return d

class RayMarcher(nn.Module):
    def __init__(self, sdf:nn.Module, max_iter:int=20):
        super(RayMarcher, self).__init__()
        self.max_iter = max_iter
        self.sdf = sdf

    def forward(self, x, u):
        d = torch.zeros_like(x)[..., :1].requires_grad_(True)
        
        for i in range(self.max_iter):
            d = d + self.sdf(x+d*u)

        return d

class OrthogonalCamera(nn.Module):
    def __init__(self, K):
        super(OrthogonalCamera, self).__init__()
        self.K = nn.parameter.Parameter(K)      # TODO : proper parametrization
    
    def forward(self, x):
        # x : (..., pose_batch, N, 3)

        return geometry._ortho_cam(x, self.K)

class PerspectiveCamera(nn.Module):
    def __init__(self, K):
        super(PerspectiveCamera, self).__init__()
        self.K = nn.parameter.Parameter(K)
    
    def forward(self, x):
        # x : (..., pose_batch, N, 3)

        return geometry._pers_cam(x, self.K)

        
class IsopointUpsampler(nn.Module):
    def __init__(self):
        super(IsopointUpsampler, self).__init__()

    def forward(self, x, y):
        # pick subset of high-metric (y) points


        # knn(set, subset)


        # upsample by interpolation


        return x


class SilhouetteSampler(nn.Module):
    def __init__(self, sdf:nn.Module, max_iter:int=20, lr:float=1e-3, sch_args:dict={}, optimizer=None, scheduler=None):
        super(SilhouetteSampler, self).__init__()
        self.sdf = sdf
        self.max_iter = max_iter
        self.lr = lr
        self.sch_args = sch_args

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR
    

    def forward(self, x, r):
        x = x.clone().detach().requires_grad_(True)
        lamb = (1e+2 * torch.ones_like(x)[..., :1]).requires_grad_(True)

        opt = self.optimizer([x, lamb], lr=self.lr)
        sch = self.scheduler(opt, **self.sch_args)
        
        for i in range(self.max_iter):
            # argmax lamb, argmin x : max(dot(dx, r), 0) + lamb * sdf(x)^2

            y, dx = utils.optimization.y_dx(x, self.sdf, create_graph=True)

            loss1 = (geometry.dot(F.normalize(dx), r)**2).mean()
            loss2 = (lamb * (y**2)).mean()

            loss = loss1 + loss2

            opt.zero_grad()
            loss.backward()
            lamb.grad *= -1     # lamb maximizes loss2
            opt.step()

            sch.step()

        return x


class SurfaceSampler(nn.Module):
    def __init__(self, sdf:nn.Module, max_iter:int=20, lr:float=1e-3, sch_args:dict={}, optimizer=None, scheduler=None):
        super(SurfaceSampler, self).__init__()
        self.sdf = sdf
        self.max_iter = max_iter
        self.lr = lr
        self.sch_args = sch_args

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.Adam

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.LinearLR
    

    def forward(self, x):
        x = x.clone().requires_grad_(True)

        opt = self.optimizer([x], lr=self.lr)
        sch = self.scheduler(opt, **self.sch_args)
        
        for i in range(self.max_iter):
            y = self.sdf(x)

            loss = (y**2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            sch.step()


class RussianRoulette(nn.Module):
    def __init__(self, p:float):
        super(RussianRoulette, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        i = torch.ones_like(x)[..., :1]
        return self.Dropout(i) * x


class PointToPixel(nn.Module):
    '''
    given a point cloud and its colors, render individual pixels in pixel space

    input : point cloud (..., n, 3), colors (..., n, k)
    output : img (..., k, h, w)
    '''
    def __init__(self, H:int, W:int, cam:nn.Module, blendmode:str = 'average'):
        super(PointToPixel, self).__init__()
        self.H = H
        self.W = W
        self.cam = cam
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

        Kx = self.cam(x)
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
    def __init__(self, cam: Optional[nn.Module]=None):
        super(PointFromPixel, self).__init__()
        self.cam = cam

    def forward(self, x, img):
        H = img.shape[-2]
        W = img.shape[-1]
        C = img.shape[-3]

        if self.cam is None:
            hH, hW = 0.5*H, 0.5*W

            K = torch.tensor(
                 [[hW, 0, hW],
                  [0, hH, hH],
                  [0, 0, 1]], 
                  dtype=torch.float, device=x.device)

            cam = lambda x: geometry._pers_cam(x, K)

        else:
            cam = self.cam

        original_shape = x.shape

        x = x.view(-1, *x.shape[-2:])
        img = img.view(-1, C, H, W)

        feat = torch.zeros((*x.shape[:-1], C), dtype=img.dtype, device=img.device)
        valid = torch.zeros((*x.shape[:-1], 1), dtype=torch.long, device=x.device)

        Kx = cam(x)
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
