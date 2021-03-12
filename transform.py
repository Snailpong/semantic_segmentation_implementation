import numpy as np
import random

class RandomCrop():
    def __init__(self, image_shape, target_shape):
        self.crop0 = random.randrange(image_shape[0]-target_shape[0])
        self.crop1 = random.randrange(image_shape[1]-target_shape[1])
        self.tar0 = target_shape[0]
        self.tar1 = target_shape[1]

    def crop_forward(self, img):
        return img[self.crop0:self.crop0+self.tar0, self.crop1:self.crop1+self.tar1]