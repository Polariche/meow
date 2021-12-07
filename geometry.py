import torch
import torch.nn as nn
import torch.nn.functional as F

def dot(x,y):
    return (x*y).sum(dim=-1, keepdim=True)

def dot2(x):
    return (x*x).sum(dim=-1, keepdim=True)

def cross(x,y):
    a_components = [x[..., 1]*y[..., 2] - x[..., 2]*y[..., 1],
                    x[..., 2]*y[..., 0] - x[..., 0]*y[..., 2],
                    x[..., 0]*y[..., 1] - x[..., 1]*y[..., 0]]

    a_components = [c.unsqueeze(-1) for c in a_components]

    a = torch.cat(a_components, dim=-1)

    return a

def length(x):
    return torch.sqrt((x**2).sum(dim=-1, keepdim=True))

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