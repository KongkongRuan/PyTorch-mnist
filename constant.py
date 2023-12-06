from torchvision import datasets,transforms
import torch
import torch.nn as nn
showImg = False
useGpu = True
EPOCH = 10
BATCH_SIZE = 20
LR = 0.001
DOWNLOAD = True
device = ''
useDevice=''
modName = ''
jmodName =''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if useGpu:
    modName = 'cnn-gpu.pt'
    jmodName = "j-cnn-gpu.pt"
    useDevice = 'gpu'
else:
    modName = 'cnn-cpu.pt'
    jmodName = "j-cnn-cpu.pt"
    useDevice = 'cpu'

train_data = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD
)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 灰度图的通道数为1
                out_channels=16,  # 16个filter
                kernel_size=5,  # filter的边长为5
                stride=1,  # 步长为1
                padding=2,  # 因为filter的边长为5，所以padding设为(5-1)/2=2
            ),
            # (16, 28, 28)
            nn.ReLU(),
            # (16, 28, 28)
            nn.MaxPool2d(
                kernel_size=2,
            ),
            # (16, 14, 14)
        )
        self.layer2 = nn.Sequential(
            # (16, 14, 14)
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            # (32, 14, 14)
            nn.ReLU(),
            # (32, 14, 14)
            nn.MaxPool2d(kernel_size=2)
            # (32, 7, 7)
        )
        self.output_layer = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # (batch, 1, 28, 28)
        x = self.layer1(x)
        # (batch, 16, 14, 14)
        x = self.layer2(x)
        # (batch, 32, 7, 7)
        x = x.reshape(x.size(0), -1)  # 将x展开为(batch, 32*7*7)
        # (batch, 32*7*7)
        output = self.output_layer(x)
        # (batch, 10)
        return output