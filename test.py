import os
import torch
import random

from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from deconvnet import DeconvNet
from fcn import FCN
from dataset import VOCSegmentationDataset

def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

if __name__ == '__main__':
    test()