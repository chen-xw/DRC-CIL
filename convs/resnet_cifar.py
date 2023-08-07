import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math

BN=nn.BatchNorm2d
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

        
class ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, scale=1, groups=1, nc=[16, 32, 64], arrange=[2,1,1]):
        super(ResNet_Cifar, self).__init__()
        self.in_planes = nc[0] * scale
        self.arrange=arrange

        self.conv1 = nn.Conv2d(3, nc[0] * scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BN(nc[0] * scale)
        self.layer1 = self._make_layer(block, nc[0] * scale , num_blocks[0], stride=1, groups=groups)
        self.layer2 = self._make_layer(block, nc[1] * scale , num_blocks[1], stride=2, groups=groups)
        self.layer3 = self._make_layer(block, nc[2] * scale , num_blocks[2], stride=2, groups=groups)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, groups=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, groups=groups, arrange=self.arrange))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out