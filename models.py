"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#### CNNs ####
class cnnSmallBN(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, width=1):
        super(cnnSmall, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channel, 32*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(32*width), 
                                      nn.ReLU(), 
                                      nn.Conv2d(32*width, 32*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(32*width), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(32*width, 64*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(64*width), 
                                      nn.ReLU(), 
                                      nn.Conv2d(64*width, 64*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(64*width), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                     )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        # Using extra linear layer helps terrifically on MNIST
        self.classifier = nn.Sequential(nn.Linear(64*width*4*4, 64*width), nn.ReLU(), nn.Linear(64*width, num_classes)) 
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
class cnnLargeBN(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, width=1):
        super(cnnLarge, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channel, 64*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(64*width), 
                                      nn.ReLU(), 
                                      nn.Conv2d(64*width, 64*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(64*width), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(64*width, 128*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(128*width), 
                                      nn.ReLU(), 
                                      nn.Conv2d(128*width, 128*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(128*width), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(128*width, 256*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(256*width), 
                                      nn.ReLU(), 
                                      nn.Conv2d(256*width, 256*width, kernel_size=3, padding=1, bias=False), 
                                      nn.BatchNorm2d(256*width), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                     )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        # Using extra linear layer helps terrifically on MNIST
        self.classifier = nn.Sequential(nn.Linear(256*width*2*2, 128*width), 
                                        nn.ReLU(), 
                                        nn.Linear(128*width, 128*width), 
                                        nn.ReLU(), 
                                        nn.Linear(128*width, num_classes)) 
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
class cnnLarge(nn.Module):
    def __init__(self, in_channel=3, num_classes=10, width=1, **kwargs):
        super(cnnLarge, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channel, 64*width, kernel_size=3, padding=1, bias=False), 
                                      nn.GroupNorm(16*width, 64*width), 
                                      nn.ReLU(), 
                                      nn.Conv2d(64*width, 64*width, kernel_size=3, padding=1, bias=False), 
                                      nn.GroupNorm(16*width, 64*width), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(64*width, 128*width, kernel_size=3, padding=1, bias=False), 
                                      nn.GroupNorm(32*width, 128*width), 
                                      nn.ReLU(), 
                                      nn.Conv2d(128*width, 128*width, kernel_size=3, padding=1, bias=False), 
                                      nn.GroupNorm(32*width, 128*width), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(128*width, 256*width, kernel_size=3, padding=1, bias=False), 
                                      nn.GroupNorm(64*width, 256*width), 
                                      nn.ReLU(), 
                                      nn.Conv2d(256*width, 256*width, kernel_size=3, padding=1, bias=False), 
                                      nn.GroupNorm(64*width, 256*width), 
                                      nn.ReLU(), 
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                     )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        # Using extra linear layer helps terrifically on MNIST
        self.classifier = nn.Sequential(nn.Linear(256*width*2*2, 128*width), 
                                        nn.ReLU(), 
                                        nn.Linear(128*width, 128*width), 
                                        nn.ReLU(), 
                                        nn.Linear(128*width, num_classes)) 
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
#### ResNet (modified for MNIST/CIFAR image size) ####
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, num_classes=10, width=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64 * width

        self.conv1 = nn.Conv2d(
            in_channel, 64*width, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64*width)
        self.layer1 = self._make_layer(block, 64*width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128*width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256*width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512*width, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512*block.expansion*width, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)