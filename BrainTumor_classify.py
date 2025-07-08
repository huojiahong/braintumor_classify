import sys
import warnings

import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    五层卷积，五次最大池化下采样224->14->7，1/2,3/5残差连接
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



if __name__ == "__main__":
    """—————————————————————————————————————————————————————————————————————
    # from torchvision import datasets 提供了很多预定义的数据集（MINIST/CIFAR-10/ImageNet等），它们已经封装好了数据加载和预处理的功能，也是基于下面的Dataset实现的
    # torchvision.datasets.ImageFolder 会自动根据文件夹名称分配类别标签，并将图像和标签加载为 PyTorch 的 Dataset 对象。
    # from torch.utils.data import Dataset 是一个抽象类，用于自定义数据集，用户需要继承类并实现__len__/__getitem__方法
    ————————————————————————————————————————————————————————————————————————"""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 确定设备对象
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # (H,W)
        # transforms.Resize(256),  # 保持图像比例，短边到256
        # transforms.CenterCrop(256),  # 经测试，有的图占比大会裁掉上下部，pass
        transforms.ToTensor(),  # 会将像素值转为float32类型并归一化到（0,1）
    ])
    ct_dataset = datasets.ImageFolder(root="./Dataset/Brain Tumor CT scan Images", transform=transform)   # image, 0-healthy/1
    """——————————————————————————————————————————————————————————————————————————
    image_tensor, label = ct_dataset[0]
    image_pil = transforms.ToPILImage()(image_tensor)
    image_pil.show()
    image_name1, label_name1 = ct_dataset.imgs[0]  # 图像的文件路径和类别标签存在datasets.ImageFolder.img中，是一个列表，每个元素是一个元组 (image_path, label)
    image_name2, label_name2 = ct_dataset.imgs[1]
    print(image_name1, label_name1)  # ./Dataset/Brain Tumor CT scan Images/Healthy/ct_healthy (1).jpg 0
    print(image_name2, label_name2)  # ./Dataset/Brain Tumor CT scan Images/Healthy/ct_healthy (1).png 0
    # sys.exit()
    ———————————————————————————————————————————————————————————————————————————————"""

    # random_split 是 PyTorch 提供的一个简单工具，用于将数据集随机划分为多个子集。
    train_ratio = 0.8
    train_size = int(train_ratio * len(ct_dataset))
    test_size = len(ct_dataset) - train_size
    train_dataset, test_dataset = random_split(ct_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=30, shuffle=True)

    net = CtModel(in_channels=3, num_classes=2)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=0.001)
    # opt = optim.SGD(params=net.parameters(), lr = 0.01, momentum=0.9)
    net = net.to(device)
    best_accuracy = float('-inf')

    from tqdm import tqdm  # 可以在循环或迭代过程中实时显示进度条
    num_epochs = 1
    ct_all_labels = []
    ct_all_predictions = []
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()  # 清空（归零）模型参数的梯度,每次调用loss.backward()时，梯度会累积到模型参数的.grad属性中，要手动清空梯度， 确保梯度是基于当前批次数据计算的
            output = net(images)  # 前向传播
            # print(output.shape, labels.shape)  # torch.Size([30, 2]) torch.Size([30])
            loss = loss_fn(output, labels)
            loss.backward()  # 反向传播
            opt.step()  # 参数更新
            epoch_loss += loss.item()

        net.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                _, predict = torch.max(output, 1)  # 沿着指定的维度返回最大值及其索引,(batch_size, num_classes)沿着类别维度计算，用于从模型的输出中获取预测的类别标签
                total += labels.shape[0]
                correct += (predict == labels).sum().item()
                if epoch == num_epochs-1:
                    ct_all_labels.extend(labels.cpu().numpy())  # 将可迭代对象的所有元素逐个添加到列表的末尾
                    ct_all_predictions.extend(predict.cpu().numpy())
            accuracy = correct / total * 100
        print(f"epoch:{epoch+1}/{num_epochs}, loss:{epoch_loss:.3f}, accuracy:{accuracy:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(net.state_dict(), 'ct_model_best.pth')
            print(f"epoch:{epoch + 1}, best model saved")
        torch.save(net.state_dict(), 'ct_model_last.pth')
        print(f"epoch:{epoch + 1}, last model saved")


    cm = confusion_matrix(ct_all_labels, ct_all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Tumor"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()






