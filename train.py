import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from tqdm import tqdm
import numpy as np

from deconvnet import DeconvNet
from fcn import FCN
from dataset import VOCSegmentationDataset

NUM_CLASSES = 21

def train():
    datasets = VOCSegmentationDataset(os.path.join(os.getcwd(), 'data', 'VOCdevkit', 'VOC2012'))
    dataloader = DataLoader(dataset=datasets[0], batch_size=32, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    net = FCN(21)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(20):
        seg_criterion = nn.NLLLoss2d(weight=class_weights)
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        net.train()
        


if __name__ == '__main__':
    train()