import torch
import torchvision.transforms.functional as F

import numbers
import random

__all__ = ["Compose", "Resize", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip"]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, fixations):
        for transform in self.transforms:
            img, fixations = transform(img, fixations)
        
        return img, fixations

class Resize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img, fixations):
        img = F.resize(img, self.size)
        fixations = F.resize(fixations, self.size)
        return img, fixations

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        target_height, target_width = output_size
        if w == target_width and h == target_height:
            return 0, 0, h, w

        i = random.randint(0, h - target_height)
        j = random.randint(0, w - target_width)

        return i, j, target_height, target_width

    def __call__(self, img, fixations):

        i, j, h, w = self.get_params(img, self.size)
        img=F.crop(img, i, j, h, w)
        fixations = F.crop(fixations, i, j, h, w)
        return img, fixations
        
class RandomHorizontalFlip(object):
    
    def __call__(self, img, fixations):
        
        if random.random() < 0.5:
            img = F.hflip(img)
            fixations = F.hflip(fixations)

        return img, fixations

class RandomVerticalFlip(object):

    def __call__(self, img, fixations):
        if random.random() < 0.5:
            img = F.vflip(img)
            fixations = F.vflip(fixations)
        
        return img, fixations

class ToTensor(object):

    def __call__(self, img, fixations):
        img = F.to_tensor(img)
        fixations = F.to_tensor(fixations)
        return img, fixations