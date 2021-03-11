from torch.utils.data import DataLoader
import os
import PIL
import torchvision.transforms as transfroms
import numpy as np
from tqdm import tqdm

NUM_CLASSES = 21

class VOCSegmentationDataset(DataLoader):
    def __init__(self, dataset_dir):
        self.files_list = os.listdir(os.path.join(dataset_dir, 'SegmentationClassAug'))

        self.image_list = []
        self.seg_mask_list = []
        self.seg_hot_list = []

        print('Data Loading...')

        for file_name in tqdm(self.files_list):
            file_name_jpg = os.path.join(dataset_dir, 'JPEGImages', file_name.split('.')[0]) + '.jpg'
            self.image_list.append(np.array(PIL.Image.open(file_name_jpg)))
            seg_original = np.array(PIL.Image.open(os.path.join(dataset_dir, 'SegmentationClassAug', file_name)))
            self.seg_mask_list.append(1 - (seg_original == 255))
            self.seg_hot_list.append(seg_original)
            break


    def __getitem__(self, index):
        seg_item = self.seg_mask_list[index]
        seg_hot_encoded = (np.arange(NUM_CLASSES) == seg_item[...,None]).astype(int)
        return self.image_list[index], seg_hot_encoded, self.seg_hot_list[index]

    def __len__(self):
        return len(self.files_list)