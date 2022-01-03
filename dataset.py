import torch

from torch.utils.data import Dataset

import pandas as pd
from torchvision.io import read_image
from typing import Union, Tuple
import transforms as T

from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

class SaliconDataset(Dataset):

    cache = []

    def __init__(self, root_dir, split, input_size: Union[int, Tuple[int, int]]):
        self.images_dir = root_dir + "/" + split
        self.fixations_dir = root_dir + "/" + split + "_fixations"
        self.input_size = input_size
        df=pd.read_csv(root_dir + "/" + split + ".csv", header=None)
        self.images_name=df[0].tolist()

    def __len__(self):
        return len(self.images_name)   

    def __getitem__(self, idx):
        

        file_name=self.images_name[idx]
        img = read_image(self.images_dir + "/" + file_name)
        fixation = read_image(self.fixations_dir + "/" + file_name)

        img, fixation = T.Resize(self.input_size).forward(img, fixation)

        c = img.shape[0]
        if c == 1:
            img = torch.cat([img, img, img], dim=0)
        
        return img, fixation
