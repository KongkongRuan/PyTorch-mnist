from torchvision import datasets,transforms
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from main import showImg
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

test_data = datasets.MNIST(
    root='./data',
    train=False
)
print(test_data.data)
print('----------------------------------------')
with torch.no_grad():
    # img = torch.autograd.Variable(torch.unsqueeze(test_data.data, dim=1))
    img = torch.unsqueeze(test_data.data, dim=1).float()

# test_x = img.type(torch.FloatTensor)[:2000] / 255  # 将将0~255压缩为0~1
test_x = img[:2000] / 255  # 将将0~255压缩为0~1
test_y = test_data.targets[:2000]




#加载现有模型
cnn=torch.load('cnn.pt')

test_output = cnn(test_x[:100])
predict_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
real_y = test_y[:100].numpy()
print(predict_y)
print(real_y)

# 打印预测和实际结果
for i in range(10):
    print('Predict', predict_y[i])
    print('Real', real_y[i])
    if showImg:
        plt.imshow(test_data.data[i].numpy(), cmap='gray')
        plt.show()