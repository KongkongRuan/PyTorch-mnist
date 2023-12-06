from torchvision import datasets,transforms
import torch
showImg = False
useGpu = False
EPOCH = 1
BATCH_SIZE = 200
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