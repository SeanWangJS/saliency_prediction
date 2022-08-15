
import argparse

from models.res_unet import ResUNet
from models.salicon import SaliconNet
from dataset import SaliconDataset

from typing import List

import torch
from torch.utils.data.dataloader import DataLoader

import transforms
import utils



device = "cuda" if  torch.cuda.is_available()  else "cpu"

batch_size=4
data_dir ="D:/data/Datasets/SALICON"

valset = SaliconDataset(data_dir, "train", (600, 448))
val_loader = DataLoader(valset, batch_size=batch_size, num_workers=0, shuffle=False)

model = SaliconNet()
model.load_state_dict(torch.load("./checkpoints/salicon-59-sb.pth"))
model.to(device)
model.eval()

transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(448),
    transforms.Normalize(mean=0.456, std=0.225)
])
criterion = utils.kld_loss2
val_loss = 0

for i, (inputs, targets) in enumerate(val_loader):
    # inputs = inputs.to(device) / 255.0
    # targets = targets.to(device) / 255.0
    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs, targets = transformer.forward(inputs, targets)
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        print(loss)
        val_loss += loss

print(val_loss / len(val_loader))