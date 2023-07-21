import torch
import torch.nn as nn

class PSNR(nn.Module):
    def __init__(self, maxi=1):
        super(PSNR, self).__init__()
        self.MAX = maxi

    def forward(self, image1, image2):
        imdff = torch.clamp(image2, 0, 1) - torch.clamp(image1, 0, 1)
        rmse = (imdff ** 2).mean().sqrt()
        ps = 20 * torch.log10(self.MAX / rmse)
        return ps