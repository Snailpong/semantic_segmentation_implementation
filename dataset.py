from torch.utils.data import DataLoader
import os
import PIL
import torchvision.transforms as transfroms
import numpy as np
from tqdm import tqdm
from transform import RandomCrop

NUM_CLASSES = 21

class VOCSegmentationDataset(DataLoader):
    def __init__(self, dataset_dir):
        self.files_list = os.listdir(os.path.join(dataset_dir, 'SegmentationClassAug'))

        self.image_list = []
        self.seg_hot_list = []

        print('Data Loading...')

        idx = 0

        for file_name in tqdm(self.files_list):
            file_name_jpg = os.path.join(dataset_dir, 'JPEGImages', file_name.split('.')[0]) + '.jpg'

            image_array = np.array(PIL.Image.open(file_name_jpg))
            if np.min(image_array.shape[:2]) < 224:
                continue

            self.image_list.append(image_array)
            seg_original = np.array(PIL.Image.open(os.path.join(dataset_dir, 'SegmentationClassAug', file_name)))
            self.seg_hot_list.append(seg_original)
            idx += 1

            if idx == 1000:
                break


    def __getitem__(self, index):
        crop = RandomCrop(self.image_list[index].shape[:2], (224, 224))
        image_item = crop.crop_forward(self.image_list[index])
        seg_hot_item = crop.crop_forward(self.seg_hot_list[index])
        return image_item, seg_hot_item

    def __len__(self):
        return 1000
        # return len(self.files_list)