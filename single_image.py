import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from constant import *



#加载现有模型

cnn=torch.load('./mod/'+useDevice+"/"+modName,map_location=device)
if useGpu:
    cnn=cnn.cuda()


transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整大小
    transforms.ToTensor(),         # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])



# 1. 加载单张图片
single_image_path = 'D:\\input\\202312041357073333333.png'  # 替换成你的图片路径
# single_image_path = 'D:\\input\\my\\4.png'  # 替换成你的图片路径
single_image = Image.open(single_image_path).convert('L')  # 转为灰度图
single_image = transform(single_image)  # 使用之前定义的预处理 transform

# 将图片添加 batch 维度
single_image = single_image.unsqueeze(0)

if showImg:
    plt.imshow(single_image.squeeze().numpy(), cmap='gray')
    plt.show()
single_image_float=single_image.float()
if useGpu:
    single_image_float=single_image_float.cuda()
# 2. 将图片传递给模型
with torch.no_grad():
    single_image_output = cnn(single_image_float)


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
