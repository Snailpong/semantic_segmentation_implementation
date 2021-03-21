from torch.utils.data import DataLoader
import os
from PIL import Image
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
            idx += 1

            file_name_jpg = os.path.join(dataset_dir, 'JPEGImages', file_name.split('.')[0]) + '.jpg'
            file_name_seg = os.path.join(dataset_dir, 'SegmentationClassAug', file_name)

            if not (os.path.isfile(file_name_jpg) and os.path.isfile(file_name_seg)):
                continue

            image_array = np.array(Image.open(file_name_jpg)) / 255.0
            if np.min(image_array.shape[:2]) <= 224:
                continue

            self.image_list.append(image_array)
            seg_original = np.array(Image.open(file_name_seg), dtype=np.int)
            self.seg_hot_list.append(seg_original)

            # if idx == 1500:
            #     break

    def __getitem__(self, index):
        crop = RandomCrop(self.image_list[index].shape[:2], (224, 224))
        image_item = crop.crop_forward(self.image_list[index])
        seg_hot_item = crop.crop_forward(self.seg_hot_list[index])

        exists_item = np.zeros(NUM_CLASSES)
        for i in range(image_item.shape[0]):
            for j in range(image_item.shape[1]):
                if 0 <= seg_hot_item[i, j] < NUM_CLASSES:
                    exists_item[seg_hot_item[i, j]] = 1 

        return image_item, seg_hot_item, exists_item

    def __len__(self):
        return len(self.image_list)