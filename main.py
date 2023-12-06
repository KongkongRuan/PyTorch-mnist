# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time



from constant import *




# DataLoader
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_data = datasets.MNIST(
    root='./data',
    train=False
)





# 为了节约时间，只使用测试集的前2000个数据
# img = Variable(
#     torch.unsqueeze(test_data.data, dim=1),
#     volatile=True
# )

# with torch.no_grad():
#     # img = torch.autograd.Variable(torch.unsqueeze(test_data.data, dim=1))
#     img = torch.unsqueeze(test_data.data, dim=1).float()
#
# # test_x = img.type(torch.FloatTensor)[:2000] / 255  # 将将0~255压缩为0~1
# test_x = img / 255  # 将将0~255压缩为0~1
# test_y = test_data.targets

# 使用所有的测试集
# test_x = Variable(
#     torch.unsqueeze(test_data.test_data, dim=1),
#     volatile=True
# ).type(torch.FloatTensor)/255 # 将将0~255压缩为0~1
cnn = CNN()
if useGpu:
    cnn = cnn.cuda() # 若有cuda环境，取消注释
#summary(cnn,input_size=(1,28,28)) #查看网络结构
#损失函数和梯度更新
#优化器
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
#损失函数
loss_func = nn.CrossEntropyLoss()

with torch.no_grad():
    # img = torch.autograd.Variable(torch.unsqueeze(test_data.data, dim=1))
    img = torch.unsqueeze(test_data.data, dim=1).float()
test_x = img / 255  # 将将0~255压缩为0~1
test_y = test_data.test_labels

if useGpu:
    test_x = test_x.cuda() # 若有cuda环境，取消注释
    test_y = test_y.cuda() # 若有cuda环境，取消注释
start=time.time()
r=range(EPOCH)
tl=enumerate(train_loader)
instart = time.time()
# 训练神经网络
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        if useGpu:
            batch_x = batch_x.cuda() # 若有cuda环境，取消注释
            batch_y = batch_y.cuda() # 若有cuda环境，取消注释
        output = cnn(batch_x)
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每隔50步输出一次信息
        if step % 50 == 0:
            test_output = cnn(test_x)
            if useGpu:
                predict_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            else:
                predict_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (predict_y == test_y).sum().item() / test_y.size(0)
            print('Epoch', epoch, '|', 'Step', step, '|', 'Loss', loss.data.item(), '|', 'Test Accuracy', accuracy)
            print(str(useDevice) + '一轮用时:' + str(time.time() - instart))
            instart = time.time()

# 预测
test_output = cnn(test_x[100:500])

if useGpu:
    predict_y = torch.max(test_output, 1)[1].cuda().data.squeeze()

else:
    predict_y = torch.max(test_output, 1)[1].data.numpy().squeeze()



print(str(device)+'总用时:'+str(time.time()-start))
if useGpu:
    test_y=test_y.cpu()
real_y = test_y[100:500].numpy()
print(predict_y)
print(real_y)

# 打印预测和实际结果
for i in range(10):
    print('Predict', predict_y[i])
    print('Real', real_y[i])
    if showImg:
        plt.imshow(test_data.data[i].numpy(), cmap='gray')
        plt.show()

cnn.eval()
torch.save(cnn, './mod/'+useDevice+"/"+modName)

#导出为traced_script格式的模型给java使用
cnn.to('cpu')
# if useGpu:
#     model.cuda()
# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 28, 28)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(cnn, example)

# 保存 TorchScript 模型
traced_script_module.save('./mod/'+useDevice+"/"+jmodName)

