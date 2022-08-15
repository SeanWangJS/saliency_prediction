from typing import Union, Tuple
import glob
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import pandas as pd


import transforms as T

class SaliconDataset(Dataset):

    def __init__(self, root_dir, split, input_size: Union[int, Tuple[int, int]]):
        self.images_dir = root_dir + "/" + split
        self.fixations_dir = root_dir + "/fixations_" + split
        self.input_size = input_size
        self.image_names = [os.path.basename(path) for path in glob.glob("{}/*.jpg".format(self.images_dir))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        file_name=self.image_names[idx]
        img = read_image(self.images_dir + "/" + file_name) / 255.0
        fixation = read_image(self.fixations_dir + "/" + file_name) / 255.0

        img, fixation = T.Resize(self.input_size).forward(img, fixation)

        c = img.shape[0]
        if c == 1:
            img = torch.cat([img, img, img], dim=0)

        return img, fixation