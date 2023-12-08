import PIL
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from constant import *
from util import convert_proba




def predictImg(path,isMy):
    # 加载现有模型

    cnn = torch.load('./mod/' + useDevice + "/" + modName, map_location=device)
    if useGpu:
        cnn = cnn.cuda()

    my = isMy

    # 1. 加载单张图片
    single_image_path = path  # 替换成你的图片路径
    # if my:
    #     single_image_path = 'D:\\input\\my\\1.png'  # 替换成你的图片路径
    # else:
    #     single_image_path = 'D:\\input\\3.png'  # 替换成你的图片路径
    single_image = Image.open(single_image_path).convert('L')  # 转为灰度图
    if my:
        single_image = PIL.ImageOps.invert(single_image)
        bbox=single_image.getbbox()
        single_image=single_image.crop(bbox)
        single_image = data_transform(single_image)  # 使用之前定义的预处理 transform
    else:
        single_image = test_transform(single_image)
    # 将图片添加 batch 维度
    single_image = single_image.unsqueeze(0)

    # if showImg:
    #     plt.imshow(single_image.squeeze().numpy(), cmap='gray')
    #     plt.show()
    single_image_float = single_image.float()
    if useGpu:
        single_image_float = single_image_float.cuda()
    # 2. 将图片传递给模型
    with torch.no_grad():
        single_image_output = cnn(single_image_float)

    tmax = torch.max(single_image_output, 1)
    one = tmax[1]
    oitem = one.item()
    # print(str(single_image_output.data))
    convert_proba(single_image_output.data)
    # 3. 获取预测结果
    single_predict_y = torch.max(single_image_output, 1)[1].item()

    # 4. 打印结果
    print('Predicted:', single_predict_y)
    if showImg:
        plt.imshow(single_image.squeeze().numpy(), cmap='gray')
        plt.show()

if __name__ == '__main__':
    isMy = False
    for i in range(10):
        if isMy:
            predictImg('D:\\input\\my\\'+str(i)+'.png',isMy)
            print("Real:"+str(i))
        else:
            predictImg('D:\\input\\' + str(i) + '.png', isMy)
            print("Real:" + str(i))

    # predictImg('D:\\input\\test\\5.png',True)
    # print("Real:" + str(5))