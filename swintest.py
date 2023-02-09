import sys

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from visdom import Visdom
from torch.autograd import Variable
import utils

from loss import SSIM,L1_Charbonnier_loss
from net import *
from data import *
parser.add_argument("--MODEL", type=str, default="", help="initial learning rate")
inputPathTest = "./dataset/test"
targePathtest = "./dataset/test_targe"
output_images = "./dataset/output/"
swinIR1 = SwinIR(upscale=1, in_chans=3, out_chans=3,img_size=128, window_size=8,
                     img_range=1., depths=[6, 6], embed_dim=360, num_heads=[8,8],
                     mlp_ratio=2, upsampler='', resi_connection='3conv')
swinIR1.eval()
swinIR1.cuda()
# 多卡
device_ids = [i for i in range(torch.cuda.device_count())]
if len(device_ids) > 1:
    swinIR1 = nn.DataParallel(swinIR1, device_ids=device_ids)
datasetTest = MyValueDataSet(inputPathTest, targePathtest)
valueLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=True, num_workers=8,
                             pin_memory=True)
psnr = utils.PSNR()
ssim = SSIM()
psnr_val_rgb = []
ssim_val_rgb = []
if os.path.exists(('./'+opt.MODEL)):
    swinIR1.load_state_dict(torch.load(('./'+opt.MODEL)))
for index, (test_j, test_ture_j) in enumerate(valueLoader, 0):
    test_j_, test_ture_j_ = test_j.cuda(), test_ture_j.cuda()
    with torch.no_grad():
        test_j_ = torch.pow(test_j_,2.2)
        test_j_ = swinIR1(test_j_)
        test_fake_j = torch.pow(test_j_, (1. / 2.2))
        save_image(test_fake_j, output_images + str(index) + ("count_J.png"))
    for output_value, target_value in zip(test_fake_j, test_ture_j_):
        psnr_val_rgb.append(psnr(output_value, target_value))
        ssim_val_rgb.append(ssim(output_value, target_value))
psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
ssim = torch.stack(ssim_val_rgb).mean().item()
print(psnr_val_rgb,ssim)
