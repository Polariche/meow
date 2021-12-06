import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import implicits
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.optim import Adam

torch.autograd.set_detect_anomaly(True)

device = "cuda"

sphere = implicits.SphereSDF(torch.tensor([0., 0., 1.]), torch.tensor([0.3])).to(device)


#sphere = implicits.SingleBVPNet(in_features=3, type='sine', num_hidden_layers=3).to(device)
optimizer = Adam([sphere.center], lr=1e-1)

raymarcher = modules.LMRayMarcher(max_iter=10)
K = torch.tensor([[128, 0, 128],
                    [0, -128, 128],
                    [0, 0, 1]], 
                    dtype=torch.float, device=device)
ptp = modules.PointToPixel(256, 256, K, blendmode='average')
pfp = modules.PointFromPixel()

target = torch.from_numpy(np.asarray(Image.open("./test_img2.png"))).permute(2,0,1)[:1].to(device)
target = target.float()/255
blurred_target = target

celoss_f = nn.BCELoss()
sigmoid_f = nn.Sigmoid()



for i in range(1):
    # sample from image
    # UV -> cam coord

    # sample (u, v) by prob = gaussian-filtered occupancy mask
    inds = torch.tensor(list(torch.utils.data.WeightedRandomSampler(blurred_target.view(-1), 5000, replacement=True)))
    H, W = target.shape[-2:]
    uv = torch.tensor([[ind / H, ind % W] for ind in inds])
    

    # sample from IF object
    # from prev.iter + gaussian perturbance

    """

    d = torch.zeros((10000,3), device=device)
    d = raymarcher(d, x, lambda x: sphere(x))

    y = sphere(d*x)
    """
