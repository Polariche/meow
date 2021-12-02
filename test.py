import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import implicits
import matplotlib.pyplot as plt


c = torch.tensor([[0., 0., 1.]])
#sphere = implicits.SphereSDF(c, torch.tensor(0.4))
sphere = implicits.Siren(in_features=3, type='sine')

x = torch.randn((100000, 3)) * 5e-1 + torch.tensor([[0., 0., 1.]])
x = F.normalize(x)
c = torch.ones((100000, 1))


d = modules.LMRayMarcher(lamb=1e-4)(torch.zeros((100000,3)), x, lambda x: sphere(x))



K = torch.tensor([[128, 0, 128],
                  [0, 128, 128],
                  [0, 0, 1]]).float()
ptp = modules.PointToPixel(256, 256, K, blendmode='min')


img1 = ptp(d*x, torch.clamp(d.detach(), 0, 1))
img2 = ptp(d*x, (sphere(d*x) < 1e-6).float())

occupancy = torch.cat(tuple(map(lambda x: x.float(), modules.PointFromPixel()(d*x, img2))), dim=-1)



print(occupancy[occupancy[:,1] == 1][:,0].mean())

plt.imsave('hi.png', img1.squeeze())
plt.imsave('occ.png', img2.squeeze())