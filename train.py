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

NUM_CLASSES = 21
BATCH_SIZE = 16

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
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # seg_criterion = nn.NLLLoss(ignore_index=255, reduction='mean')
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    # cls_criterion = nn.BCEWithLogitsLoss()

    cur_iters = 0

    for epoch in range(30):
        model.train()

        pbar = tqdm(range(len(dataloader)))
        pbar.set_description('Epoch {}'.format(epoch+1))

        total_loss = 0.

        for idx, (images, hots) in enumerate(dataloader):
            cur_iters += 1

            images = images.to(device, dtype=torch.float32).permute(0, 3, 1, 2)
            hots = hots.to(device, dtype=torch.long)

            optimizer.zero_grad()
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