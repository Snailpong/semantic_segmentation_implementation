import os
import torch
import random

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from tqdm import tqdm
import numpy as np

from deconvnet import DeconvNet
from fcn import FCN
from dataset import VOCSegmentationDataset

NUM_CLASSES = 21
BATCH_SIZE = 32

def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    dataset = VOCSegmentationDataset(os.path.join(os.getcwd(), 'data', 'VOCdevkit', 'VOC2012'))
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    model = DeconvNet(21)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #seg_criterion = nn.NLLLoss2d()
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    # cls_criterion = nn.BCEWithLogitsLoss()

    cur_iters = 0

    for epoch in range(20):
        model.train()

        pbar = tqdm(range(len(dataloader)))

        for (images, masks, hots) in dataloader:
            cur_iters += 1

            # print(images.shape, hots.shape)

            images = images.to(device, dtype=torch.float32).permute(0, 3, 1, 2)
            masks = masks.to(device, dtype=torch.int).permute(0, 3, 1, 2)
            hots = hots.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)

            # output_mask = (outputs * masks).squeeze()
            # hots = hots.squeeze()

            # print(outputs.shape, hots.shape)

            # loss = seg_criterion(output_mask, hots)
            loss = seg_criterion(outputs, hots)
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str('loss: ' + str(np.around(loss.detach().cpu().numpy(), 4)))
            pbar.update()

        


if __name__ == '__main__':
    train()