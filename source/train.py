# load RGB image & masks


## sampling stage ##
# sample from occupancy masks

# sample from implicit


# pass RGB image through CNN for features & camera pose


# evaluate with DIF; minimize BCE occupancy on samples

import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import implicits
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.optim import Adam
import utils
import geometry
import neural_implicits
import utils.optimization
import json


with open('./setups/setup1.json') as f:
    args = json.load(f)

N = args['number_of_point_samples']
L = args['render_img_length']


device = "cuda"


sphere = neural_implicits.SingleBVPNet(in_features=3, type='sine', num_hidden_layers=3).to(device)
optimizer = Adam(sphere.parameters(), lr=args['learning_rate'])

raymarcher = modules.LMRayMarcher(max_iter=args['raymarcher_max_iter'], sdf=sphere)
K = torch.tensor([[L/2, 0, L/2],
                    [0, L/2, L/2],
                    [0, 0, 1]], 
                    dtype=torch.float, device=device)
K_inv = torch.inverse(K)

cam = modules.PerspectiveCamera(K)
ptp = modules.PointToPixel(L, L, cam, blendmode='average')
pfp = modules.PointFromPixel()


celoss_f = nn.BCEWithLogitsLoss()
sigmoid_f = nn.Sigmoid()


# load target img
target = torch.from_numpy(np.asarray(Image.open("./data/toy/test_img2.png"))).permute(2,0,1)[:1].to(device)
target = target.float()/255

# initialize obj iso-points
x_obj_sample_prev = torch.randn((N, 2), device=device)
x_obj_sample_prev = torch.cat([x_obj_sample_prev, torch.ones_like(x_obj_sample_prev)[..., :1]], dim=-1)


x0 = None

for i in range(100):
    # sample from image
    # UV -> cam coord

    with torch.no_grad():
        # sample (u, v) by prob
        inds = torch.tensor(list(torch.utils.data.WeightedRandomSampler(target.view(-1), N, replacement=True)))
        H, W = target.shape[-2:]
        uv = torch.tensor([[ind % H, ind // H] for ind in inds], device=device)

        # convert to world
        uv1 = torch.cat([uv, torch.ones_like(uv)[..., :1]], dim=-1).float()
        x_img_sample = torch.matmul(uv1, K_inv.transpose(-1,-2))

        # gaussian perturbance
        x_img_sample += torch.randn(x_img_sample.shape, device=device) * args['point_sample_perturb']


        # sample from estimated locations
        # from prev.iter + gaussian perturbance
        x_obj_sample = x_obj_sample_prev + torch.randn(x_obj_sample_prev.shape, device=device) * args['point_sample_perturb']


    r = torch.cat([x_img_sample, x_obj_sample], dim=0)
    r = r / r[..., -1:]

    x0 = torch.zeros_like(r)

    #d = raymarcher(x0, r).unsqueeze(-2)
    d = torch.randn(*r.shape[:-1], args['number_of_dist_samples'], 1, device=r.device) * args['dist_perturb'] + 1
    x = x0.unsqueeze(-2)+d*r.unsqueeze(-2)

    
    y, dx = utils.optimization.y_dx(x, sphere)

    argmin_ind = torch.argmin(torch.sign(y) * d, dim=-2)


    x = torch.gather(x, -2, torch.cat([argmin_ind.unsqueeze(-1)]*3, -1))[..., 0, :] 
    y = torch.gather(y, -2, argmin_ind.unsqueeze(-1))[..., 0, :]
    dx = torch.gather(dx, -2, torch.cat([argmin_ind.unsqueeze(-1)]*3, -1))[..., 0, :]

    # losses
    occ_pts = -y / 1e-3
    tar_pts = pfp(r, target)[0]

    bce_pts = F.binary_cross_entropy_with_logits(occ_pts, tar_pts)
    reg = (torch.norm(dx, dim=-1) - 1).pow(2).mean()

    loss = bce_pts + reg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



