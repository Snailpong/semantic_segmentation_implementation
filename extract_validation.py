import os
import random

if __name__ == '__main__':
    random.seed(1234)

    dataset_dir = os.getcwd() + '/data/VOCdevkit/VOC2012'
    os.makedirs(dataset_dir + '/JPEGImagesVal', exist_ok=True)
    files_list = os.listdir(os.path.join(dataset_dir, 'JPEGImages'))
    file_available_list = []

    for file_name in files_list:
        file_name_jpg = os.path.join(dataset_dir, 'JPEGImages', file_name)
        file_name_seg = os.path.join(dataset_dir, 'SegmentationClassAug', file_name.split('.')[0] + '.png')

        if os.path.isfile(file_name_jpg) and os.path.isfile(file_name_seg):
            file_available_list.append(file_name)
            
    files_val_list = random.sample(file_available_list, len(file_available_list) // 5)

    for file_name in files_val_list:
        file_name_seg_src = os.path.join(dataset_dir, 'JPEGImages', file_name)
        file_name_seg_tar = os.path.join(dataset_dir, 'JPEGImagesVal', file_name)

        os.rename(file_name_seg_src, file_name_seg_tar)

