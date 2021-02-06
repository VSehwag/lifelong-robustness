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
