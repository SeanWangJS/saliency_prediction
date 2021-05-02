import torch
import json
import torchvision.transforms as transforms

import PIL.Image as Image
import glob
from torch.utils.data import Dataset

import pandas as pd

import os

class SaliconDataset(Dataset):
    def __init__(self, images_dir, fixations_dir, image_names_csv, transformer):
        self.images_dir = images_dir
        self.fixations_dir = fixations_dir
        self.transformer = transformer
        df=pd.read_csv(image_names_csv, header=None)
        self.images_name=df[0].tolist()

    def __len__(self):
        return len(self.images_name)   
    
    def __getitem__(self, idx):
        file_name=self.images_name[idx]
        img=Image.open(self.images_dir + "/" + file_name)
        fixations = Image.open(self.fixations_dir + "/" + file_name)

        img, fixations = self.transformer(img, fixations)
        c, _, _ = img.shape
        if c == 1:
            img = torch.cat([img, img, img], dim=0)
        
        return img, fixations