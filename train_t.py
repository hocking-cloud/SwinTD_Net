import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from visdom import Visdom

import utils
from loss import SSIM,L1_Charbonnier_loss
from net import SwinIR
from data import *
inputPathTrain_T = './dataset/gamma'
train_dcp_a = '.dataset/DCP_train/dcp_a'
targetPathTrain = './dataset/traintarge'
parser = argparse.ArgumentParser()
parser.add_argument("--EPOCH",type=int,default=2000,help="starting epoch")
parser.add_argument("--BATCH_SIZE",type=int,default=8,help="size of the batches")
parser.add_argument("--PATCH_SIZE",type=int,default=64,help="size of the patch")
parser.add_argument("--LEARNING_RATE",type=float,default=2e-5,help="initial learning rate")
opt = parser.parse_args()

criterion2 = L1_Charbonnier_loss().cuda()
criterion1 = SSIM().cuda()
#实例化数据加载器及模型
swinIR1 = SwinIR(upscale=1, in_chans=3,out_chans=1, img_size=opt.PATCH_SIZE, window_size=8,
                      img_range=1., depths=[6, 6], embed_dim=360, num_heads=[8,8],
                      mlp_ratio=2, upsampler='', resi_connection='3conv')

swinIR1.cuda()
device_ids = [i for i in range(torch.cuda.device_count())]
if len(device_ids) > 1:
    swinIR1 = nn.DataParallel(swinIR1, device_ids=device_ids)
optimizer = optim.AdamW(swinIR1.parameters(), lr=opt.LEARNING_RATE)
datasetTrain = MyTrainDataSet_T(inputPathTrain_T,targetPathTrain,train_dcp_a ,patch_size=opt.PATCH_SIZE)
trainLoader = DataLoader(dataset=datasetTrain, batch_size=opt.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=8,
                         pin_memory=True)

# # 实例化窗口
wind = Visdom()
# # 初始化窗口信息
wind.line([0.],[0.], win = 'epochloss', opts = dict(title = 'epochloss'))
print('-------------------------------------------------------------------------------------------------------')
for epoch in range(opt.EPOCH):
    swinIR1.train(True)
    #进度条
    iters = tqdm(trainLoader, file=sys.stdout)
    epochLoss = 0
    timeStart = time.time()
    for index, (gamma_hazy,dcp_a,targe) in enumerate(iters, 0):

        swinIR1.zero_grad()
        optimizer.zero_grad()
        #包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息
        gamma_hazy,dcp_a,targe= Variable(gamma_hazy).cuda(),Variable(dcp_a).cuda(),Variable(targe).cuda()
        fake_t  = swinIR1(gamma_hazy)
        fake_t = torch.clamp(fake_t,max=(1-1e-6),min=0)
        one = Variable(torch.ones(opt.BATCH_SIZE,1,))
        fake_hazy = targe*fake_t+dcp_a*torch.sub(one,fake_t)
        hazy = torch.pow(gamma_hazy,(1/2.2))
        loss2 = criterion2(fake_hazy, hazy)
        loss1 = criterion1(fake_hazy, hazy)
        loss = loss2+(1-loss1)
        loss.backward()
        optimizer.step()
        epochLoss += loss.item()
        #进度条
        iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, opt.EPOCH, loss.item()))
    print(epochLoss)
    wind.line([epochLoss], [epoch], win='epochloss', update='append')
    torch.save(swinIR1.state_dict(), 'model_T_best.pth')

