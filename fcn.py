import torch
from torch import nn
from torch.nn import functional as F

class ConvLayer1(nn.Module):
  def __init__(self, channel_input, channel_output):
    super(ConvLayer1, self).__init__()
    self.conv1 = nn.Conv2d(channel_input, channel_output, 3, padding=1)
    self.conv2 = nn.Conv2d(channel_output, channel_output, 3, padding=1)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2)

  def forword(self, x):
    x = self.conv1(x)
    x = self.relu()
    x = self.conv2(x)
    x = self.relu()
    return self.pool(x)


class ConvLayer2(nn.Module):
  def __init__(self, channel_input, channel_output):
    super(ConvLayer2, self).__init__()
    self.conv1 = nn.Conv2d(channel_input, channel_output, 3, padding=1)
    self.conv2 = nn.Conv2d(channel_output, channel_output, 3, padding=1)
    self.conv3 = nn.Conv2d(channel_output, channel_output, 3, padding=1)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2)
    
  def forword(self, x):
    x = self.conv1(x)
    x = self.relu()
    x = self.conv2(x)
    x = self.relu()
    x = self.conv3(x)
    x = self.relu()
    return self.pool(x)

  
class ConvLayer3(nn.Module):
  def __init__(self, n_class):
    super(ConvLayer3, self).__init__()
    self.conv1 = nn.Conv2d(512, 4096, 7)
    self.conv2 = nn.Conv2d(4096, 4096, 1)
    self.conv3 = nn.Conv2d(4096, n_class, 1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu()
    x = self.conv2(x)
    x = self.relu()
    x = self.conv3(x)
    return x
  
  
# VGG16, FCN-8s

class FCN(nn.Module):
  def __init__(self, n_class):
    super(FCN, self).__init__()
    self.layer1 = ConvLayer1(3, 64)
    self.layer2 = ConvLayer1(64, 128)
    self.layer3 = ConvLayer2(128, 256)
    self.layer4 = ConvLayer2(256, 512)
    self.layer5 = ConvLayer2(512, 512)
    self.layer6 = ConvLayer3(n_class)

    self.conv16s = nn.Conv2d(512, n_class, 1)
    self.conv8s = nn.Conv2d(256, n_class, 1)

  def forward(self, x):
    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    x5 = self.layer5(x4)
    x6 = self.layer5(x5)

    x6_upsampled = F.upsample(x6, scale_factor=2, mode='bilinear')
    fcn_16s = self.conv16s(x5) + x6_upsampled
    fcn_16s_upsampled = F.upsample(fcn_16s, scale_factor=2, mode='bilinear')
    fcn_8s = self.conv8s(x4) + fcn_16s_upsampled
    
    return F.upsample(fcn_8s, scale_factor=8, mode='bilinear')
    
