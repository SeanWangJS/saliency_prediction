import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
from torch.utils.data.dataloader import DataLoader

import transforms
from dataset import SaliconDataset
from models.salicon import SaliconNet
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(448),
     transforms.Normalize(mean=0.456, std=0.225)
])

trainset = SaliconDataset(root_dir="./dataset_example", split="train2014examples", input_size=(600, 448))
trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

model = SaliconNet()
model.eval()

model.to(device)

i, (inputs, labels) = next(enumerate(trainloader))
inputs = inputs.to(device)
labels = labels.to(device)
inputs, labels = transformer.forward(inputs, labels)
with torch.no_grad():
    outputs=model(inputs)

loss = utils.kld_loss(outputs, labels)
print(loss)