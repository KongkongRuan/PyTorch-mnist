import torch
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
# from main import showImg

showImg = False
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
#加载现有模型
cnn=torch.load('cnn.pt')


transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整大小
    transforms.ToTensor(),         # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])



# 1. 加载单张图片
# single_image_path = 'D:\\input\\202312041357073333333.png'  # 替换成你的图片路径
single_image_path = 'D:\\input\\4.png'  # 替换成你的图片路径
single_image = Image.open(single_image_path).convert('L')  # 转为灰度图
single_image = transform(single_image)  # 使用之前定义的预处理 transform

# 将图片添加 batch 维度
single_image = single_image.unsqueeze(0)

if showImg:
    plt.imshow(single_image.squeeze().numpy(), cmap='gray')
    plt.show()

# 2. 将图片传递给模型
with torch.no_grad():
    single_image_output = cnn(single_image.float())


tmax=torch.max(single_image_output, 1)
one=tmax[1]
oitem=one.item()
print(str(single_image_output.data))
# 3. 获取预测结果
single_predict_y = torch.max(single_image_output, 1)[1].item()

# 4. 打印结果
print('Predicted:', single_predict_y)
if showImg:
    plt.imshow(single_image.squeeze().numpy(), cmap='gray')
    plt.show()
