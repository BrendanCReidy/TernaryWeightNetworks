'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from twn import *


class TernaryLeNet5(nn.Module):
    def __init__(self, ternarized=True):
        super(TernaryLeNet5,self).__init__()
        self.conv1 = TernaryConv2d(1,32,kernel_size = 5, ternarized=ternarized)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = TernaryConv2d(32,64,kernel_size = 5, ternarized=ternarized)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = TernaryLinear(1600,512, ternarized=ternarized)
        self.fc2 = TernaryLinear(512,10, ternarized=ternarized)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.bn_conv1(x),2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.bn_conv2(x),2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  

class TernaryBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, ternarized=False):
        super(TernaryBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = TernaryConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, ternarized=ternarized)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = TernaryConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False, ternarized=ternarized)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = TernaryConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, ternarized=ternarized)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.ternarized = ternarized

        self.shortcut = nn.Sequential()
        if in_planes != out_planes and self.stride==1:
            skip_connection = TernaryConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, ternarized=ternarized)
            skip_connection.meta_data.append("residual")
            self.shortcut = nn.Sequential(
                skip_connection,
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride==1:
            out = out + self.shortcut(x)
        return out


class TernaryMobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, ternarized=True):
        super(TernaryMobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.ternarized = ternarized

        self.conv1 = TernaryConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, ternarized=ternarized)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = TernaryConv2d(320, 1260, kernel_size=1, stride=1, padding=0, bias=False, ternarized=ternarized)
        self.bn2 = nn.BatchNorm2d(1260)
        self.linear = TernaryLinear(1260, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(TernaryBlock(in_planes, out_planes, expansion, stride, ternarized=self.ternarized))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    net = TernaryMobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
