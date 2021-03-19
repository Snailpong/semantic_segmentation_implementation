import torch
import torchvision.models as models
from torch import nn
from torch.nn import functional as F

class ResNet18(nn.Module):
    FEAT = 512
    FEAT_4 = 256

    def __init__(self):
        super(ResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        self.layer3[0].conv1.stride = (1,1)
        self.layer3[0].downsample[0].stride = (1,1)
        self.layer3[1].conv1.dilation = (2,2)
        self.layer3[1].conv1.padding = (2,2)
        self.layer3[1].conv2.dialation = (2,2)
        self.layer3[1].conv1.padding = (2,2)

        self.layer4[0].conv1.stride = (1,1)
        self.layer4[0].downsample[0].stride = (1,1)
        self.layer4[1].conv1.dilation = (4,4)
        self.layer4[1].conv1.padding = (4,4)
        self.layer4[1].conv2.dialation = (4,4)
        self.layer4[1].conv1.padding = (4,4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x, x_aux


class PSPModule(nn.Module):
    def __init__(self):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self.make_stage(size) for size in (1, 2, 3, 6)])
        self.conv = nn.Conv2d(ResNet18.FEAT * 5, ResNet18.FEAT, 1)
        self.relu = nn.ReLU()

    def make_stage(self, size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(ResNet18.FEAT, ResNet18.FEAT, 1, bias=False)
        return nn.Sequential(pool, conv)
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        stages = [F.interpolate(stage(x), size=(h, w), mode='bilinear') for stage in self.stages] + [x]
        x = torch.cat(stages, dim=1)
        x = self.conv(x)
        x = self.relu(x)
        return x


class UpsampleModule(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(UpsampleModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_input, channel_output, 3, padding=1),
            nn.BatchNorm2d(channel_output),
            nn.PReLU())
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv(x)
        return x


class PSPNet(nn.Module):
    def __init__(self, n_class):
        super(PSPNet, self).__init__()
        self.bottleneck = ResNet18()
        self.pspModule = PSPModule()
        self.upsample = nn.Sequential(UpsampleModule(ResNet18.FEAT, 256), UpsampleModule(256, 64), UpsampleModule(64, 64))
        self.final = nn.Sequential(nn.Conv2d(64, n_class, 1), nn.LogSoftmax())
        self.classifier = nn.Sequential(nn.Linear(ResNet18.FEAT_4, 256), nn.ReLU(), nn.Linear(256, n_class))

    def forward(self, x):
        x, x_aux = self.bottleneck(x)
        x = self.pspModule(x)
        x = self.upsample(x)
        x = self.final(x)
        x_aux = F.adaptive_max_pool2d(x_aux, output_size=(1, 1)).view(-1, x_aux.size(1))
        x_aux = self.classifier(x_aux)
        return x, x_aux