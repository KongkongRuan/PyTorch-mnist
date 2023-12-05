import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
EPOCH = 2
BATCH_SIZE = 100
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

for i in range(2):
    print(train_data.targets[i].item())
    plt.imshow(train_data.data[i].numpy(),cmap='gray')
    plt.show()
