import sys
import argparse
import torch.nn.functional as  F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from visdom import Visdom
from ptflops import get_model_complexity_info
import utils
from loss import SSIM, L1_Charbonnier_loss
from net import SwinIR,Generator
from data import *

inputPathTrain_j = '/data/dahazingnet/dataset/inputtrain'  # 输入的训练文件j
targetPathTrain = '/data/dahazingnet/dataset/targettrain_i'  # 输入训练集的无雾图像

inputPathTest = "/data/dahazingnet/dataset/inputtest"
targePathtest = "/data/dahazingnet/dataset/targettest_i"
imagename = os.listdir(inputPathTest)

parser = argparse.ArgumentParser()
parser.add_argument("--EPOCH", type=int, default=1, help="starting epoch")
parser.add_argument("--BATCH_SIZE", type=int, default=32, help="size of the batches")
parser.add_argument("--PATCH_SIZE", type=int, default=64, help="size of the patch")
parser.add_argument("--LEARNING_RATE", type=float, default=2e-6, help="initial learning rate")
opt = parser.parse_args()

# 实例化
psnr = utils.PSNR()
criterion1 = SSIM().cuda()
criterion2 = L1_Charbonnier_loss().cuda()
# 实例化数据加载器及模型
swinIR2 = SwinIR(upscale=1, in_chans=3, out_chans=3,img_size=128, window_size=8,
                     img_range=1., depths=[6, 6], embed_dim=360, num_heads=[8,8],
                     mlp_ratio=2, upsampler='', resi_connection='3conv')

swinIR2.cuda()
device_ids = [i for i in range(torch.cuda.device_count())]
if len(device_ids) > 1:
    swinIR2 = nn.DataParallel(swinIR2, device_ids=device_ids)
optimizer = torch.optim.Adam([
            {'params': swinIR2.parameters(),"lr" : 2e-5},
        ])
torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200],gamma=0.1)
datasetTrain = TrainDataSet_J(inputPathTrain_j, targetPathTrain, patch_size=opt.PATCH_SIZE)
trainLoader = DataLoader(dataset=datasetTrain, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=8,drop_last=True,
                         pin_memory=True)
# # 实例化窗口
wind = Visdom()
# # 初始化窗口信息
wind.line([0.],[0.], win = 'l1', opts = dict(title = 'l1'))
wind.line([0.],[0.], win = 'ssim', opts = dict(title = 'ssim'))


print('-------------------------------------------------------------------------------------------------------')
for epoch in range(opt.EPOCH):
    swinIR2.train(True)
    # 进度条
    iters = iters(trainLoader, file=sys.stdout)
    epochLoss1 = 0
    epochLoss2 = 0
    tureloss = 0
    for index, (j, ture_j) in enumerate(iters, 0):
        swinIR2.zero_grad()
        optimizer.zero_grad()
        # 包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息
        j_, ture_j_ = Variable(j).cuda(), Variable(ture_j).cuda()
        j2 = swinIR2(j_)
        loss1 = criterion1(j2, ture_j_)
        loss2 = criterion2(j2, ture_j_)

        loss = (1-loss1) + loss2
        loss.backward()
        optimizer.step()
        # 进度条
        epochLoss1 +=loss1.item()
        epochLoss2 +=loss2.item()
        iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss1 %.6f,batch Loss2 %.6f' % (epoch + 1, opt.EPOCH, (1-loss1.item()),loss2.item()))
    wind.line([epochLoss1], [epoch], win='ssim', update='append')
    wind.line([epochLoss2], [epoch], win='l1', update='append')
    print(epochLoss1,epochLoss2)
    torch.save(swinIR2.state_dict(), './nogama.pth')

