import click
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

from dataset import VOCSegmentationDataset
from util import getColorMap, segmentationColorize, getModel


NUM_CLASSES = 21


@click.command()
@click.option('--model_name', default='pspnet')
@click.option('--image_path', default='./data/VOCdevkit/VOC2012/JPEGImages/2007_007109.jpg')
def test(model_name, image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    os.makedirs('./result', exist_ok=True)

    model = getModel(model_name, NUM_CLASSES)
    model.to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'model', '{}.pth'.format(model.__class__.__name__))))
    else:
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'model', '{}.pth'.format(model.__class__.__name__)), map_location=torch.device('cpu')))
    model.eval()
    
    image = Image.open(image_path)
    image_size = image.size

    image = image.resize((224, 224), Image.BILINEAR)
    image = ToTensor()(image)
    image = torch.unsqueeze(image, 0)

    if model_name == 'pspnet': output, _ = model(image)
    else: output = model(image)

    output = torch.argmax(torch.squeeze(output), dim=0)
    output = Image.fromarray(np.array(output, dtype=np.uint8), 'L')
    output = np.array(output.resize(image_size, Image.NEAREST))

    colorMap = getColorMap('./colorMap.txt')
    
    outputColorMap = segmentationColorize(output, colorMap)
    outputColorMap = Image.fromarray(outputColorMap, 'RGB')
    outputColorMap.save('./result/{}_result.jpg'.format('.'.join(os.path.basename(image_path).split('.')[:-1])))


if __name__ == '__main__':
    test()