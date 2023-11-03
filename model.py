# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = self.pool3(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
