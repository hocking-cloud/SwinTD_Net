import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from visdom import Visdom

import utils
from loss import SSIM, L1_Charbonnier_loss,VGGLoss
from net import SwinIR
from data import *

inputPathTrain_j = '/data/dahazingnet/dataset/DCP_train/dcp_j'  # 输入的训练文件j
targetPathTrain = '/data/dahazingnet/dataset/targettrain_i'  # 输入训练集的无雾图像
inputPathTest_j = '/data/dahazingnet/dataset/DCP_test/dcp_j'
targetPathTest = '/data/dahazingnet/dataset/targettest_i'
output_images = "/data/dahazingnet/dataset/output_image/"
imagename = os.listdir(inputPathTest_j)

parser = argparse.ArgumentParser()
parser.add_argument("--EPOCH", type=int, default=500, help="starting epoch")
parser.add_argument("--BATCH_SIZE", type=int, default=10, help="size of the batches")
parser.add_argument("--PATCH_SIZE", type=int, default=96, help="size of the patch")
parser.add_argument("--LEARNING_RATE", type=float, default=2e-4, help="initial learning rate")
opt = parser.parse_args()

datasetTrain = MyTrainDataSet_J(inputPathTrain_j, targetPathTrain, patch_size=opt.PATCH_SIZE)
trainLoader = DataLoader(dataset=datasetTrain, batch_size=opt.BATCH_SIZE, shuffle=False, num_workers=8,drop_last=True,
                         pin_memory=True)
datasetValue = MyValueDataSet(inputPathTest_j, targetPathTest)
valueLoader = DataLoader(dataset=datasetValue, batch_size=1, shuffle=False, drop_last=True, num_workers=8,
                           pin_memory=False)
datamean = [0,0,0]
for index, (j, ture_j) in enumerate(trainLoader, 0):
    print(index)
    for i in  range(3):
        datamean[i] += j[:,i,:,:].mean()
print(np.array(datamean)/1399)