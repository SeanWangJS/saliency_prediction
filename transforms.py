import torch
import torchvision.transforms.functional as F

import numbers
import random

__all__ = ["Compose", "Resize", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip"]

class Compose(torch.nn.Module):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def forward(self, img, fixations):
        for transform in self.transforms:
            img, fixations = transform.forward(img, fixations)
        
        return img, fixations

class Resize(torch.nn.Module):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def forward(self, img, fixations):
        img = F.resize(img, self.size)
        fixations = F.resize(fixations, self.size)
        return img, fixations

class RandomCrop(torch.nn.Module):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size()[-2:]
        target_height, target_width = output_size
        if w == target_width and h == target_height:
            return 0, 0, h, w

        i = random.randint(0, h - target_height)
        j = random.randint(0, w - target_width)

        return i, j, target_height, target_width

    def forward(self, img, fixations):

        i, j, h, w = self.get_params(img, self.size)
        img=F.crop(img, i, j, h, w)
        fixations = F.crop(fixations, i, j, h, w)
        return img, fixations
        
class RandomHorizontalFlip(torch.nn.Module):
    
    def forward(self, img, fixations):
        
        if random.random() < 0.5:
            img = F.hflip(img)
            fixations = F.hflip(fixations)

        return img, fixations

class RandomVerticalFlip(torch.nn.Module):

    def forward(self, img, fixations):
        if random.random() < 0.5:
            img = F.vflip(img)
            fixations = F.vflip(fixations)
        
        return img, fixations

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def forward(self, imgs, fixations):
        imgs = F.normalize(imgs, self.mean, self.std)
        return imgs, fixations

class ToTensor(object):

    def forward(self, img, fixations):
        img = F.to_tensor(img)
        fixations = F.to_tensor(fixations)
        return img, fixations