import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import os


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(ResidualBlock, self).__init__()
        self.fx = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
        )
        if in_channels != out_channels:  # 层内的残差，featuremap大小不变，只改变通道
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fx(x)
        identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class CtModel(nn.Module):
    """
    五层卷积，四次最大池化下采样224->14，1/2,3/5残差连接
    """
    def __init__(self, in_channels, num_classes):
        super(CtModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # n,c,224,224->n,64,112,112
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # n,64,112,112->n,128,56,56
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # n,128,56,56->n,256,28,28
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # n,256,28,28->n,512,14,14
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        ) # n,512,14,14->n,1024,14,14
        self.downsample = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=1, stride=2), # n,256,28,28和n,1024,14,14融合
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=7*7*1024, out_features=512),
            nn.ReLU(),
            # nn.Dropout(0.5),  # Dropout to prevent overfitting
            nn.Linear(in_features=512, out_features=num_classes)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        single = x
        x = self.layer4(x)
        x = self.layer5(x)
        x = x + self.downsample(single)
        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x


class LoadImageTensor:
    def __init__(self, path):
        if os.path.isdir(path):
            file = [os.path.join(path, filename) for filename in os.listdir(path) if filename.lower().endswith(('.jpg', 'jpeg', '.png', '.webp'))]
        else:
            file = [path]
        self.file = file
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.index = 0

    def __getitem__(self, idx):
        filename = self.file[idx]
        image_idx = Image.open(filename)
        image_idx = self.transform(image_idx)
        image_idx = torch.unsqueeze(image_idx, dim=0)
        return filename, image_idx

    # def __len__(self):
    #     return len(self.file)
    #
    # def __iter__(self):
    #     self.index = 0
    #     return self
    #
    # def __next__(self):
    #     if self.index >= len(self.file):
    #         raise StopIteration
    #     image_idx = self[self.index]  # 调用getitem方法
    #     self.index += 1
    #     return image_idx


net = CtModel(in_channels=3, num_classes=2)
net.load_state_dict(torch.load('ct_model_best.pth',  map_location=torch.device('cpu')))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image_path = 'eval_images'
eval_images = LoadImageTensor(image_path)
result = {0:'health', 1:'tumor'}
for name, image in eval_images:
    logit = net(image)
    _, predict = torch.max(logit, 1)  # 取每行最大值和索引
    predict = result[predict.numpy()[0]]
    print(name, predict)
    # eval_images/2.jpeg tumor
    # eval_images/3.jpg tumor


# net.eval()
# example = torch.randn(1, 3, 224, 224)
# torch.onnx.export(
#     model=net,
#     args=example,
#     f='BrainTumor_CTclassify.onnx',
#     input_names=['images'],
#     output_names=['labels'],
#     opset_version=12,
#     dynamic_axes={
#         'images': {
#             0: 'batch'
#         },
#         'labels': {
#             0: 'batch'
#         }
#     }  # 给定是否是动态结构
# )
#
# # https://netron.app可查网络结构图