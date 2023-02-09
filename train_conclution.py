import sys

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from visdom import Visdom
from torch.autograd import Variable
from net import *
from loss import SSIM,L1_Charbonnier_loss
from net import Generator
from data import *


def train_conclution():

    EPOCH = 1000
    BATCH_SIZE = 8
    PATCH_SIZE = 64#(image_size)
    LEARNING_RATE = 2e-4
    LEARNING_RATE2 = 2e-6
    LEARNING_RATE3 = 2e-7
    eps = 1e-7
    lr_list = []
    loss_list = []

    inputPathTrain = './dataset/gamama'
    train_dcp_a = './dataset/DCP_train/dcp_a'
    train_dcp_T = './dataset/DCP_train/dcp_t'



    criterion1 = SSIM().cuda()
    criterion2 = L1_Charbonnier_loss().cuda()
    gen = Generator(n_residual_blocks=12)
    gen.cuda()
    swinIR2 = SwinIR(upscale=1, in_chans=3, out_chans=3, img_size=64, window_size=8,
                     img_range=1., depths=[6, 6], embed_dim=360, num_heads=[8, 8],
                     mlp_ratio=2, upsampler='', resi_connection='3conv')
    swinIR1 = SwinIR(upscale=1, in_chans=3, out_chans=1, img_size=64, window_size=8,
                     img_range=1., depths=[6, 6], embed_dim=360, num_heads=[8, 8],
                     mlp_ratio=2, upsampler='', resi_connection='3conv')


    device_ids = [i for i in range(torch.cuda.device_count())]
    if len(device_ids) > 1:
        swinIR1 = nn.DataParallel(swinIR1, device_ids=device_ids)
        swinIR2 = nn.DataParallel(swinIR2, device_ids=device_ids)
        gen = nn.DataParallel(gen, device_ids=device_ids)
    optimizer = torch.optim.Adam([
            {'params': gen.parameters(), 'lr': LEARNING_RATE, },
            {'params': swinIR2.parameters(),"lr" : LEARNING_RATE2},
            {'params': swinIR1.parameters(), "lr": LEARNING_RATE3},
        ])
    #训练数据
    datasetTrain = TrainDataSet(inputPathTrain,train_dcp_a,train_dcp_T,patch_size=64)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8,
                             pin_memory=True)

    # # 实例化窗口
    wind = Visdom()
    # # 初始化窗口信息
    wind.line([0.], [0.], win='psnr', opts=dict(title='psnr'))
    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists('./genbest.pth'):
        print(1111)
        gen.load_state_dict(torch.load('./genbest.pth'))
    for epoch in range(EPOCH):
        gen.train(True)
        swinIR2.train(True)
        swinIR2.train(True)
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss1 = 0
        epochLoss2 = 0
        epochLoss3 = 0
        for index, (gamma_hazy,dcp_a,targe_t) in enumerate(iters, 0):

            gen.zero_grad()
            swinIR1.zero_grad()
            swinIR2.zero_grad()
            optimizer.zero_grad()
            gamma_hazy, dcp_a, targe_t = Variable(gamma_hazy).cuda(), Variable(dcp_a).cuda(), Variable(targe_t).cuda()
            fake_t = swinIR1(gamma_hazy)
            fake_j = swinIR2(gamma_hazy)
            fake_t = torch.clamp(fake_t, max=(1 - 1e-6), min=0)
            one = Variable(torch.ones(BATCH_SIZE, 1, ))
            fake_j = torch.pow(fake_j,(1/2.2))
            fake_hazy = fake_j * fake_t + dcp_a * torch.sub(one, fake_t)
            hazy = torch.pow(gamma_hazy,2.2)
            loss1 = criterion2(fake_hazy,hazy)
            loss2 = criterion1(fake_hazy,hazy)
            loss3 = criterion2(fake_t,targe_t)
            loss = (loss1 + loss3 )+(1-loss2)
            loss.backward()
            optimizer.step()

            #wind.line([loss4.item()], [index], win='psnr', update='append')
            iters.set_description('L1 %.3f L2 %.3f L3 %.3fL2 %.3f'%(loss1.item(),loss2.item(),loss3.item()))
        torch.save(gen.state_dict(), 'genbest.pth')
        torch.save( swinIR2.state_dict(), 'swinir1best.pth')
        # if epoch>50:
train_conclution()

