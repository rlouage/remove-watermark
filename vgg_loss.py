import torchvision
import torch.nn as nn
import torch

class VGG19(nn.Module):
    def __init__(self, requires_grad=False, criterion=nn.L1Loss()):
        super(VGG19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained = True, progress = True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.criterion = criterion
        for i in range(2):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(2, 7):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(7, 12):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(12, 21):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])
        for i in range(21, 30):
            self.slice5.add_module(str(i), vgg_pretrained_features[i])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x1 = self.slice1(x)
        x2 = self.slice2(x1)
        x3 = self.slice3(x2)
        x4 = self.slice4(x3)
        x5 = self.slice5(x4)
        return [x1, x2, x3, x4, x5]
    
    def calc_loss(self, x, y):
        outx = self.forward(x)
        outy = self.forward(y)
        loss = 0
        
        for i in range(len(outx)):
            _,_,w,h = outx[i].shape
            loss += self.criterion(outx[i], outy[i].detach())/(w*h)
        return loss


class VGG11(nn.Module):
    def __init__(self, requires_grad=False, criterion=nn.L1Loss()):
        super(VGG11, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg11(pretrained = True, progress = True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.criterion = criterion
        for i in range(2):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(2, 8):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(8, 13):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(13, 18):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])
        for i in range(18, 20):
            self.slice5.add_module(str(i), vgg_pretrained_features[i])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        x1 = self.slice1(x)
        x2 = self.slice2(x1)
        x3 = self.slice3(x2)
        x4 = self.slice4(x3)
        x5 = self.slice5(x4)
        return [x1, x2, x3, x4, x5]
    
    def calc_loss(self, x, y):
        outx = self.forward(x)
        outy = self.forward(y)
        loss = 0
        
        for i in range(len(outx)):
            _,_,w,h = outx[i].shape
            loss += self.criterion(outx[i], outy[i].detach())/(w*h)
        return loss
