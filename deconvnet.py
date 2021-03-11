import torch
from torch import nn
from torch.nn import functional as F

class ConvLayer1(nn.Module):
    def __init__(self, channel_input, channel_output):
        self.conv1 = nn.Conv2d(channel_input, channel_output, 3, padding=1)
        self.batch1 = nn.BatchNorm2d(channel_output)
        self.conv2 = nn.Conv2d(channel_output, channel_output, 3, padding=1)
        self.batch2 = nn.BatchNorm2d(channel_output)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        return self.pool(x)


class ConvLayer2(nn.Module):
    def __init__(self, channel_input, channel_output):
        self.conv1 = nn.Conv2d(channel_input, channel_output, 3, padding=1)
        self.batch1 = nn.BatchNorm2d(channel_output)
        self.conv2 = nn.Conv2d(channel_output, channel_output, 3, padding=1)
        self.batch2 = nn.BatchNorm2d(channel_output)
        self.conv3 = nn.Conv2d(channel_output, channel_output, 3, padding=1)
        self.batch3 = nn.BatchNorm2d(channel_output)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu(x)
        return self.pool(x)
      
      
class DeconvLayer1(nn.Module):
    def __init__(self, channel_input, channel_output):
        self.unpool = nn.MaxUnpool2d(2)
        self.deconv1 = nn.ConvTranspose2d(channel_input, channel_input, 3, padding=1)
        self.batch1 = nn.BatchNorm2d(channel_input)
        self.deconv2 = nn.ConvTranspose2d(channel_input, channel_output, 3, padding=1)
        self.batch2 = nn.BatchNorm2d(channel_output)
        self.relu = nn.ReLU()

    def forward(self, x, indicate):
        x = self.unpool(x, indicate)
        x = self.deconv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.batch2(x)
        return self.relu(x)


class DeconvLayer2(nn.Module):
    def __init__(self, channel_input, channel_output, indicate):
        self.indicate = indicate
        self.unpool = nn.MaxUnpool2d(2)
        self.deconv1 = nn.ConvTranspose2d(channel_input, channel_input, 3, padding=1)
        self.batch1 = nn.BatchNorm2d(channel_input)
        self.deconv2 = nn.ConvTranspose2d(channel_input, channel_input, 3, padding=1)
        self.batch2 = nn.BatchNorm2d(channel_input)
        self.deconv3 = nn.ConvTranspose2d(channel_input, channel_output, 3, padding=1)
        self.batch3 = nn.BatchNorm2d(channel_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.unpool(x, indicate)
        x = self.deconv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.batch3(x)
        return self.relu(x)

      
class FCLayer(nn.Module):
    def __init__(self):
        self.fc1 = nn.Conv2d(512, 4096, 7, padding=0)
        self.batch1 = nn.BatchNorm2d(4096)
        self.fc2 = nn.Conv2d(4096, 4096, 1, padding=0)
        self.batch2 = nn.BatchNorm2d(4096)
        self.fc3 = nn.ConvTranspose2d(4096, 512, 7)
        self.batch3 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch3(x)
        return self.relu(x)
      
      
# input_size = (224, 224, 3)

class DeconvNet(nn.Module):
    def __init__(self, n_class):
        self.convLayer1 = ConvLayer1(3, 64)
        self.convLayer2 = ConvLayer1(64, 128)
        self.convLayer3 = ConvLayer2(128, 256)
        self.convLayer4 = ConvLayer2(256, 512)
        self.convLayer5 = ConvLayer2(512, 512)
        self.fcLayer = FCLayer()
        self.deconvLayer5 = DeconvLayer2(512, 512)
        self.deconvLayer4 = DeconvLayer2(512, 256)
        self.deconvLayer3 = DeconvLayer2(256, 128)
        self.deconvLayer2 = DeconvLayer1(128, 64)
        self.deconvLayer1 = DeconvLayer1(64, 64)
        self.segScore = nn.Conv2d(64, n_class, 1, padding=0)

    def forward(self, x):
        x, i1 = self.convLayer1(x)
        x, i2 = self.convLayer2(x)
        x, i3 = self.convLayer3(x)
        x, i4 = self.convLayer4(x)
        x, i5 = self.convLayer5(x)
        x = self.fcLayer(x)
        x = self.deconvLayer5(x, i5)
        x = self.deconvLayer4(x, i4)
        x = self.deconvLayer3(x, i3)
        x = self.deconvLayer2(x, i2)
        x = self.deconvLayer1(x, i1)
        return self.segScore(x)
