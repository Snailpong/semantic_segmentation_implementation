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
        self.files_list = []
        self.dataset_dir = dataset_dir
        files_all_list = os.listdir(os.path.join(dataset_dir, 'SegmentationClassAug'))
        
        for file_name in files_all_list:

            file_name_jpg = os.path.join(dataset_dir, 'JPEGImages', file_name.split('.')[0]) + '.jpg'
            file_name_seg = os.path.join(dataset_dir, 'SegmentationClassAug', file_name)

            if os.path.isfile(file_name_jpg) and os.path.isfile(file_name_seg):
                self.files_list.append(file_name)


    def __getitem__(self, index):
        file_name = self.files_list[index]

        file_name_jpg = os.path.join(self.dataset_dir, 'JPEGImages', file_name.split('.')[0]) + '.jpg'
        file_name_seg = os.path.join(self.dataset_dir, 'SegmentationClassAug', file_name)

        img_original = np.array(Image.open(file_name_jpg)) / 255.0
        seg_original = np.array(Image.open(file_name_seg), dtype=np.int)

        if img_original.shape[0] < 224 or img_original.shape[1] < 224:
            H, W, _ = img_original.shape

            img_offset = np.zeros((max(H, 224), max(W, 224), 3))
            seg_offset = np.full((max(H, 224), max(W, 224)), 255)
            img_offset[0:H, 0:W] = img_original
            seg_offset[0:H, 0:W] = seg_original
            img, seg = img_offset, seg_offset

        else:
            img, seg = img_original, seg_original

        crop = RandomCrop(img.shape[:2], (224, 224))
        image_item = crop.crop_forward(img)
        seg_hot_item = crop.crop_forward(seg)

        exists_item = np.zeros(NUM_CLASSES)
        for i in range(NUM_CLASSES):
            if np.any(image_item == i):
                exists_item[i] = 1
                    
        return image_item, seg_hot_item, exists_item


    def __len__(self):
        return len(self.files_list)