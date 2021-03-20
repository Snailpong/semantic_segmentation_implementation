import os
import torch
import random
from PIL import Image

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.autograd import Variable

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from deconvnet import DeconvNet
from fcn import FCN
from pspnet import PSPNet
from dataset import VOCSegmentationDataset
from util import getColorMap, segmentationColorize

def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    os.makedirs('./result', exist_ok=True)

    model = PSPNet(21)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'model', '{}.pth'.format(model.__class__.__name__)), map_location=torch.device('cpu')))
    
    model.eval()
    
    image = Image.open('./data/VOCdevkit/VOC2012/JPEGImages/2007_007109.jpg')
    image_size = image.size
    image.save('./result/input.jpg')

    seg_true = np.array(Image.open('./data/VOCdevkit/VOC2012/SegmentationClassAug/2007_007109.png'))

    image = image.resize((224, 224), Image.BILINEAR)
    image = ToTensor()(image)
    image = torch.unsqueeze(image, 0)

    output, _ = model(image)
    output = torch.argmax(torch.squeeze(output), dim=0)
    output = Image.fromarray(np.array(output, dtype=np.uint8), 'L')
    output = output.resize(image_size, Image.NEAREST)
    output = np.array(output)

    colorMap = getColorMap('./colorMap.txt')
    
    outputColorMap = segmentationColorize(output, colorMap)
    outputColorMap = Image.fromarray(outputColorMap, 'RGB')
    outputColorMap.save('./result/output2.jpg')

    outputColorMap = segmentationColorize(seg_true, colorMap)
    outputColorMap = Image.fromarray(outputColorMap, 'RGB')
    outputColorMap.save('./result/output_true.jpg')


if __name__ == '__main__':
    test()