import os
import random
import time
import torchvision
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image
import natsort

from torchvision.utils import save_image
class TrainDataSet(Dataset):
    def __init__(self, inputPathTrain_j,inputPathTrain_a,inputPathTrain_t,patch_size=128,dataname="indoor"):
        super(TrainDataSet, self).__init__()
        #文件路径
        self.inputPath_j = inputPathTrain_j
        self.inputPath_a = inputPathTrain_a
        self.inputPath_t = inputPathTrain_t
        #获取文件路径的所有文件
        self.inputImages_j = natsort.natsorted(os.listdir(inputPathTrain_j),alg=natsort.ns.PATH)
        self.inputImages_a = natsort.natsorted(os.listdir(inputPathTrain_a),alg=natsort.ns.PATH)
        self.inputImages_t = natsort.natsorted(os.listdir(inputPathTrain_t),alg=natsort.ns.PATH)

        #图片尺寸
        self.ps = patch_size
        if dataname=="indoor":
            self.lable=10
        elif dataname=="outdoor":
            self.lable=35
        else:
            self.lable=1
    #获取文件数量
    def __len__(self):
        return len(self.inputPathtrains)
    #当对象是序列时，键是整数。当对象是映射时（字典），键是任意值。
    def __getitem__(self, index):
        #以rgb形式读取图片
        ps = self.ps
        index1 = index//self.lable

        inputImagePath_j = os.path.join(self.inputPath_j, self.inputImages_j[index])
        inputImage_j = Image.open(inputImagePath_j).convert('RGB')
        inputImagePath_a = os.path.join(self.inputPath_a, self.inputImages_a[index])
        inputImage_a = torch.load(inputImagePath_a)
        inputImagePath_t = os.path.join(self.inputPath_t, self.inputImages_t[index])
        inputImage_t = Image.open(inputImagePath_t).convert('RGB')
        targetPathtrain_t = os.path.join(self.targetPathtrain_t, self.targetPathtrains_t[index])
        targetImage_t = Image.open(targetPathtrain_t).convert('L')
        #转换成张量
        inputImage_j = ttf.to_tensor(inputImage_j)

        targetImage_t = ttf.to_tensor(targetImage_t)

        inputImage_a = torch.mean(inputImage_a,dim=0).view(1,1,1)


        hh, ww = inputImage_j.shape[1], inputImage_j.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        #批量获取张量
        input_j = inputImage_j[:, rr:rr+ps, cc:cc+ps]
        target_t = targetImage_t[:, rr:rr + ps, cc:cc + ps]

        return input_j, inputImage_a,target_t
class TrainDataSet_T(Dataset):
    def __init__(self, inputPathTrain_j,inputPathTrain_a,inputPathTrain_t, targetPathTrain,inputaPathtrain,targetPathtrain_t,patch_size=128,dataname="indoor"):
        super(TrainDataSet, self).__init__()
        #文件路径
        self.inputPath_j = inputPathTrain_j
        self.inputPath_a = inputPathTrain_a
        self.inputPath_t = inputPathTrain_t
        #获取文件路径的所有文件
        self.inputImages_j = natsort.natsorted(os.listdir(inputPathTrain_j),alg=natsort.ns.PATH)
        self.inputImages_a = natsort.natsorted(os.listdir(inputPathTrain_a),alg=natsort.ns.PATH)
        self.inputImages_t = natsort.natsorted(os.listdir(inputPathTrain_t),alg=natsort.ns.PATH)
        #同上
        self.targetPath = targetPathTrain
        self.targetImages = natsort.natsorted(os.listdir(targetPathTrain),alg=natsort.ns.PATH)
        self.inputPathtrain = inputaPathtrain
        self.inputPathtrains = natsort.natsorted(os.listdir(inputaPathtrain),alg=natsort.ns.PATH)
        self.targetPathtrain_t = targetPathtrain_t
        self.targetPathtrains_t = natsort.natsorted(os.listdir(targetPathtrain_t),alg=natsort.ns.PATH)
        #图片尺寸
        self.ps = patch_size
        if dataname=="indoor":
            self.lable=10
        elif dataname=="outdoor":
            self.lable=35
        else:
            self.lable=1
    #获取文件数量
    def __len__(self):
        return len(self.inputPathtrains)
    #当对象是序列时，键是整数。当对象是映射时（字典），键是任意值。
    def __getitem__(self, index):
        #以rgb形式读取图片
        ps = self.ps
        index1 = index//self.lable

        inputImagePath_j = os.path.join(self.inputPath_j, self.inputImages_j[index])
        inputImage_j = Image.open(inputImagePath_j).convert('RGB')
        inputImagePath_a = os.path.join(self.inputPath_a, self.inputImages_a[index])
        inputImage_a = torch.load(inputImagePath_a)
        targetImagePath = os.path.join(self.targetPath, self.targetImages[index1])
        targetImage = Image.open(targetImagePath).convert('RGB')
        #转换成张量
        inputImage_j = ttf.to_tensor(inputImage_j)
        targetImage = ttf.to_tensor(targetImage)

        inputImage_a = torch.mean(inputImage_a,dim=0).view(1,1,1)

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        #批量获取张量
        input_j = inputImage_j[:, rr:rr+ps, cc:cc+ps]
        target = targetImage[:, rr:rr+ps, cc:cc+ps]
        return input_j, inputImage_a, target,
class MyValueDataSet(Dataset):
    def __init__(self, inputPathTest_j,targetPathTest):
        super(MyValueDataSet, self).__init__()
        #文件路径
        self.inputPath_j = inputPathTest_j
        #获取文件路径的所有文件
        self.inputImages_j = natsort.natsorted(os.listdir(inputPathTest_j), alg=natsort.ns.PATH)
        #同上
        self.targetPath = targetPathTest
        self.targetImages = natsort.natsorted(os.listdir(targetPathTest), alg=natsort.ns.PATH)
    def __len__(self):
        return len(self.inputImages_j)
    def __getitem__(self, index):
        #以rgb形式读取图片
        index1 = index//10
        inputImagePath_j = os.path.join(self.inputPath_j, self.inputImages_j[index])
        inputImage_j = Image.open(inputImagePath_j).convert('RGB')
        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')
        inputImage_j = ttf.to_tensor(inputImage_j)
        targetImage = ttf.to_tensor(targetImage)
        ps = 64

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        #批量获取张量
        # inputImage_j = inputImage_j[:, rr:rr+ps, cc:cc+ps]
        # targetImage = targetImage[:, rr:rr+ps, cc:cc+ps]
        return inputImage_j, targetImage
class OneDataSet(Dataset):
    def __init__(self,inputpath):
        super(OneDataSet, self).__init__()

        #文件路径
        self.inputPath = inputpath
        #获取文件路径的所有文件
        self.inputImages = natsort.natsorted(os.listdir(inputpath), alg=natsort.ns.PATH)
    def __len__(self):
        return len(self.inputImages)
    def __getitem__(self, index):
        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')
        #转换成张量
        inputImage = ttf.to_tensor(inputImage)
        return inputImage
class TrainDataSet_J(Dataset):
    def __init__(self, inputPathTrain_j, targetPathTrain,patch_size=96,dataname="indoor"):
        super(TrainDataSet_J, self).__init__()
        #文件路径
        self.inputPath_j = inputPathTrain_j
        #获取文件路径的所有文件
        self.inputImages_j = natsort.natsorted(os.listdir(inputPathTrain_j), alg=natsort.ns.PATH)
        #同上
        self.targetPath = targetPathTrain

        self.targetImages = natsort.natsorted(os.listdir(targetPathTrain), alg=natsort.ns.PATH)

        #图片尺寸
        self.ps = patch_size
        if dataname=="indoor":
            self.lable=10
        elif dataname=="outdoor":
            self.lable=35
        else:
            self.lable=1
    #获取文件数量
    def __len__(self):
        return len(self.inputImages_j)
    #当对象是序列时，键是整数。当对象是映射时（字典），键是任意值。
    def __getitem__(self, index):
        #以rgb形式读取图片
        ps = self.ps
        index1 = index//self.lable
        inputImagePath_j = os.path.join(self.inputPath_j, self.inputImages_j[index])
        inputImage_j = Image.open(inputImagePath_j).convert('RGB')
        targetImagePath = os.path.join(self.targetPath, self.targetImages[index1])
        targetImage = Image.open(targetImagePath).convert('RGB')
        #转换成张量
        inputImage_j = ttf.to_tensor(inputImage_j)
        targetImage = ttf.to_tensor(targetImage)

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        #批量获取张量
        input_j = inputImage_j[:, rr:rr+ps, cc:cc+ps]
        target = targetImage[:, rr:rr+ps, cc:cc+ps]
        return input_j , target
