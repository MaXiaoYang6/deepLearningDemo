import torch.nn as nn
import  torch.nn.functional as F

class LeNet(nn.Module):
    def __int__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) # deep = 3 out_channels = 16 kernel_size = 5  output = 16 * 28 * 28
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x) # output(16, 14, 14)
        x = F.relu(self.conv2(2)) # output(32, 10, 10)
        x = self.pool2(x) # output(32, 5, 5)
        x = x.view(-1, 32*5*5) # output(32*5*5)
        x = F.relu(self.fc1(x)) # output(120)
        x = F.relu(self.fc2(x)) # output(84)
        x = self.fc3(x) # output(10)
        return x

# 经卷积后的矩阵尺寸大小计算公式为
# N = (W - F + 2P) / S + 1   (32 - 5 + 0) / 1 + 1 = 28

# 输入图片大小为 W * W    32 * 32
# Filter (卷积核大小)大小为 F * F   5 * 5
# padding 的像素数为 P

import torch
input1 = torch.rand([32, 3, 32, 32])
model = LeNet()
print(input1)
print(model)
output = model(input1)