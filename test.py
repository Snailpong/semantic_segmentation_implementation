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
@click.option('--image_path', default='./data/VOCdevkit/VOC2012/JPEGImagesVal')
@click.option('--evaluate', type=click.BOOL, default=True)
def test(model_name, image_path, evaluate):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    evaluate_list = []
    os.makedirs('./result', exist_ok=True)

    model = getModel(model_name, NUM_CLASSES)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'model', '{}.pth'.format(model.__class__.__name__)), map_location=device))
    model.eval()

    colorMap = getColorMap('./colorMap.txt')

    if os.path.isdir(image_path):
        files_list = []
        file_names_list = os.listdir(image_path)
        for file_name in file_names_list:
            files_list.append(os.path.join(image_path, file_name))
    else:
        files_list = [image_path]

    for idx, file_path in enumerate(files_list):
        file_name = '.'.join(os.path.basename(file_path).split('.')[:-1])
        print('\r{}/{} {}'.format(idx, len(files_list), file_name), end=' ')

        image = Image.open(file_path)
        image_size = image.size

        image = image.resize((224, 224), Image.BILINEAR)
        image = ToTensor()(image)
        image = torch.unsqueeze(image, 0)

        if model_name == 'pspnet': output, _ = model(image)
        else: output = model(image)

        output = torch.argmax(torch.squeeze(output), dim=0)
        output = Image.fromarray(np.array(output, dtype=np.uint8), 'L')
        output = np.array(output.resize(image_size, Image.NEAREST))

        if evaluate:
            ground_truth_path = './data/VOCdevkit/VOC2012/SegmentationClassAug/{}.png'.format(file_name)
            ground_truth = np.array(Image.open(ground_truth_path), dtype=np.int)
            ground_truth[ground_truth == 255] = 0
            output_evaluate = output.copy()
            output_evaluate[np.where(ground_truth == 0)] = 0

            union = ground_truth | output_evaluate
            intersection = ground_truth & output_evaluate

            iou = intersection.sum() / union.sum()
            print(iou, end='')
            evaluate_list.append(iou)

        outputColorMap = segmentationColorize(output, colorMap)
        outputColorMap = Image.fromarray(outputColorMap, 'RGB')
        outputColorMap.save('./result/{}.jpg'.format(file_name))

    if evaluate:
        print()
        print(np.mean(evaluate_list), np.std(evaluate_list))


if __name__ == '__main__':
    test()