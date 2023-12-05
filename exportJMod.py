import torch
import torchvision
import torch.nn as nn
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载模型，我这里是保存的模型全部参数，所以直接load，如果只保存了state_dict，需要其他写法，这里不做说明.
model = torch.load('./cnn.pt', map_location=device)

model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 1, 28, 28)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# 保存 TorchScript 模型
traced_script_module.save("model-j.pt")
