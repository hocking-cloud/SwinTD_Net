from data import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.utils import save_image

class Net(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, image1):
        img  = torch.pow(image1,2.2)
        return img
"########################################################################"
inputPathTrain= './dataset/inputtrain'
gamatrain = "./dataset/gamma"
datasetTest = OneDataSet(inputPathTrain)
testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=1,
                        pin_memory=True)
net = Net().cuda
device_ids = [i for i in range(torch.cuda.device_count())]
if len(device_ids) > 1:
    net = nn.DataParallel(net, device_ids=device_ids)
i = 1

for index, x in enumerate(testLoader):
    x = x.cuda()
    with torch.no_grad():
        output_test = net(x)
        save_image(output_test, inputPathTrain + str(index)+str(".png"))