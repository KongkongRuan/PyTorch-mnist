import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision import datasets,transforms
import matplotlib.pyplot as plt


showImg = False
EPOCH = 20
BATCH_SIZE = 200
LR = 0.001
DOWNLOAD = True

train_data = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD
)
print(train_data.data.size())
print(train_data.targets.size())

for i in range(200):
    print(train_data.targets[i].item())
    if showImg:
        plt.imshow(train_data.data[i].numpy(),cmap='gray')
        plt.show()
