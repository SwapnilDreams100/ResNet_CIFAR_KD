import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_res(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_res, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # Forming F
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #Adding skip connection
        out += self.shortcut(x)

        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, P, N, C1, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        stride = [1, 2, 2, 2]
        C = C1
        self.in_planes = C
        self.N = N
        self.P = P
        self.conv1 = nn.Conv2d(3, C, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.layers1 = nn.ModuleList()
        for i in range(0,N):
          if i>0:
            C = 2*C
          self.layers1.append(self._make_layer(block, C, num_blocks[i], stride[i]))

        self.linear = nn.Linear(C, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(0,self.N):
            out = self.layers1[i](out)
        out = F.avg_pool2d(out, self.P)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def project1_model():
    return ResNet(4, 4, 40, BasicBlock_res, [5,4,2,2])
