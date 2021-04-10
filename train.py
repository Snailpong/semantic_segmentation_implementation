import click
import os
import torch
import random
import time

from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from dataset import VOCSegmentationDataset
from util import getModel


NUM_CLASSES = 21
BATCH_SIZE = 16


@click.command()
@click.option('--model_name', default='pspnet')
def train(model_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    dataset = VOCSegmentationDataset(os.path.join(os.getcwd(), 'data', 'VOCdevkit', 'VOC2012'))
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    model = getModel(model_name, NUM_CLASSES)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    cls_criterion = nn.BCEWithLogitsLoss()

    cur_iters = 0

    for epoch in range(50):
        model.train()

        pbar = tqdm(range(len(dataloader)))
        pbar.set_description('Epoch {}'.format(epoch+1))

        total_loss = 0.

        for idx, (images, hots, exists) in enumerate(dataloader):
            images = images.to(device, dtype=torch.float32).permute(0, 3, 1, 2)
            hots = hots.to(device, dtype=torch.long)
            exists = exists.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            if model_name == 'pspnet':
                outputs, outputs_cls = model(images)
                seg_loss = seg_criterion(outputs, hots)
                cls_loss = cls_criterion(outputs_cls, exists)
                loss = seg_loss + 0.4 * cls_loss
            else: 
                outputs = model(images)
                loss = seg_criterion(outputs, hots)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            pbar.set_postfix_str('loss: ' + str(np.around(total_loss / (idx + 1), 4)))
            pbar.update()

        torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model', '{}.pth'.format(model.__class__.__name__)))


if __name__ == '__main__':
    train()