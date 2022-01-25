import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import math
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        # added for hsic trainining
        if isinstance(x, tuple):
            x, output_list = x
        else:
            output_list = []
        # added ends
        
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        
        output_list.append(out)
        return out, output_list


class NetworkBlock(nn.Module):
    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth=34,
        num_classes=10,
        widen_factor=10,
        dropRate=0.0,
        rob= False,
        shrink = 1
    ):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        nChannels = [int(k*shrink) for k in nChannels]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )
        # 1st sub-block
        self.sub_block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, dropRate
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, dropRate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, dropRate
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.rob = rob
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        output_list = []
        
        out = self.conv1(x)
        output_list.append(out)
        
        out, out_list = self.block1(out)
        output_list.extend(out_list)
        
        out, out_list = self.block2(out)
        output_list.extend(out_list)
        
        out, out_list = self.block3(out)
        output_list.extend(out_list)
        
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        output_list.append(out)
        
        out = self.fc(out)
        if self.rob:
            return out
        else:
            return out, output_list
        
    
def WideResNet28_10(**kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
    shrink = kwargs['shrink'] if 'shrink' in kwargs else 1
    return WideResNet(28, num_classes, 10, rob=rob, shrink=shrink)

def WideResNet28_4(**kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
    shrink = kwargs['shrink'] if 'shrink' in kwargs else 1
    return WideResNet(28, num_classes, 4, rob=rob, shrink=shrink)

def WideResNet34_10(**kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    num_classes = kwargs['num_classes'] if 'num_classes' in kwargs else 10
    shrink = kwargs['shrink'] if 'shrink' in kwargs else 1
    return WideResNet(34, num_classes, 10, rob=rob, shrink=shrink)


if __name__ == '__main__':
    model=WideResNet(34, 10, 10, rob=True, shrink=0.35)
    
    dense_params = 48262586
    total_params = 0
    for name, weight in model.named_parameters():
        if weight.requires_grad:
            total_params += weight.numel()
            print(name, weight.numel())
    print("{} params in total".format(total_params))
    print("{} sparsity comparing to dense model".format(total_params/dense_params))
    