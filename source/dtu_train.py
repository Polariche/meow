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
from scene_dataset import SceneDataset
from torch.utils.data import DataLoader
import math 

torch.autograd.set_detect_anomaly(True)

with open('./setups/setup1.json') as f:
    args = json.load(f)

N = args['number_of_point_samples']
L = args['render_img_length']


device = "cuda"


sphere = neural_implicits.SingleBVPNet(in_features=3, type='sine', num_hidden_layers=3).to(device)
optimizer = Adam(sphere.parameters(), lr=args['learning_rate'])

raymarcher = modules.RayMarcher(max_iter=args['raymarcher_max_iter'], sdf=sphere)


celoss_f = nn.BCEWithLogitsLoss()
sigmoid_f = nn.Sigmoid()

# initialize obj iso-points
x_obj_sample_prev = torch.randn((N, 2), device=device)
x_obj_sample_prev = torch.cat([x_obj_sample_prev, torch.ones_like(x_obj_sample_prev)[..., :1]], dim=-1)

scenes = DataLoader(SceneDataset(False, 'DTU', (L, L), args['scan_num']), batch_size=args['batch_size'], shuffle=True)
iterator = iter(scenes)

_, samples, _ = next(iterator)
targets = samples['object_mask'].unsqueeze(1).float().to(device)

for i in range(1000):
    # sample from image
    # UV -> cam coord

    #_, samples, _ = next(iterator)
    #targets = samples['object_mask']

    with torch.no_grad():
        x_img_sample = []
        for target in targets:
                
            # sample (u, v) by prob
            inds = torch.tensor(list(torch.utils.data.WeightedRandomSampler(target.float().view(-1), N, replacement=True)))
            H, W = target.shape[-2:]
            uv = torch.tensor([[ind % H, ind // H] for ind in inds], device=device)

            target_K = torch.tensor([[W/2, 0, W/2],
                                    [0, H/2, H/2],
                                    [0, 0, 1]], 
                                    dtype=torch.float, device=device)
            target_K_inv = torch.inverse(target_K)

            # convert to world
            uv1 = torch.cat([uv, torch.ones_like(uv)[..., :1]], dim=-1).float()
            b_x_img_sample = torch.matmul(uv1, target_K_inv.transpose(-1,-2))

            # gaussian perturbance
            b_x_img_sample += torch.randn(b_x_img_sample.shape, device=device) * args['point_sample_perturb']
            x_img_sample.append(b_x_img_sample)

        x_img_sample = torch.cat(x_img_sample, dim=0)


        # sample from estimated locations
        # from prev.iter + gaussian perturbance

        x_obj_sample = x_obj_sample_prev + torch.randn(x_obj_sample_prev.shape, device=device) * args['point_sample_perturb']


        r = torch.cat([x_img_sample, torch.cat([x_obj_sample] * args['batch_size'], dim=0)], dim=-2)
        r = r / r[..., -1:]

        x0 = torch.zeros_like(r)

        cam = modules.PerspectiveCamera(target_K)
        ptp = modules.PointToPixel(H, W, cam, blendmode='average')
        pfp = modules.PointFromPixel()

    d = torch.randn(*r.shape[:-1], args['number_of_dist_samples'], 1, device=r.device) * args['dist_perturb'] + 1
    x = x0.unsqueeze(-2)+d*r.unsqueeze(-2)

    
    y, dx = utils.optimization.y_dx(x, sphere)

    argmin_ind = torch.argmin(torch.sign(y) * d, dim=-2)


    x = torch.gather(x, -2, torch.cat([argmin_ind.unsqueeze(-1)]*3, -1))[..., 0, :] 
    y = torch.gather(y, -2, argmin_ind.unsqueeze(-1))[..., 0, :]
    dx = torch.gather(dx, -2, torch.cat([argmin_ind.unsqueeze(-1)]*3, -1))[..., 0, :]

    # losses
    occ_pts = -y / 1e-3 #sigmoid_f(- y / 1e-3)

    

    tar_pts = pfp(r, target)[0]

    bce_pts = F.binary_cross_entropy_with_logits(occ_pts, tar_pts)
    reg = (torch.norm(dx, dim=-1) - 1).pow(2).mean()

    loss = bce_pts + reg

    #l2_pts = (sigmoid_f(occ_pts) - tar_pts).pow(2).sum(dim=-1).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # visualization
    
    occ = ptp(r, occ_pts)
    tar = ptp(r, tar_pts)

    bce = F.binary_cross_entropy_with_logits(occ, tar)
    bce_dx = torch.autograd.grad(bce, [occ], retain_graph=True, create_graph=False)[0]

    plt.imsave('results/occ_%d.png' % i, sigmoid_f(occ).squeeze().detach().cpu().numpy())
    plt.imsave('results/tar_%d.png' % i, tar.squeeze().detach().cpu().numpy())

    #plt.imsave('results/l2_occ_%d.png' % i, 2 * (tar - occ).squeeze().detach().cpu().numpy())
    plt.imsave('results/bce_occ_%d.png' % i, - bce_dx.squeeze().detach().cpu().numpy())

    visible_pts_inds = (occ_pts > 0.5).view(-1)
    x_visible = x[visible_pts_inds]

