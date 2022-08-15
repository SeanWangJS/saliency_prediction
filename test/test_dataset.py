import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from dataset import SaliconDataset

dataset_dir = "./dataset_example"

dataset=SaliconDataset(dataset_dir, "train2014examples", (448, 448))
print(len(dataset))
img, fixation = dataset[0]

print(img.shape)
print(fixation.shape)
    