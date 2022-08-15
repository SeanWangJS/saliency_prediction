import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.io import read_image
from torchvision import transforms as T

from dataset import SaliconDataset
from models.salicon import SaliconNet, QuantizableSaliconNet
import utils
import transforms

import numpy as np

device = "cpu"

model = QuantizableSaliconNet()
model.load_state_dict(torch.load("./checkpoints/salicon-salicon-new2-59.pth"))
model.to(device)
model.eval()

## fuse model
model.fuse_model()

## setting qconfig
backend = "fbgemm"
model.qconfig = torch.quantization.get_default_qconfig(backend)

## prepare
qmodel = torch.quantization.prepare(model, inplace=False)

## calibrate
data_dir = "D:/data/Datasets/SALICON"
valset = SaliconDataset(data_dir, "val", (800, 600))
val_loader = DataLoader(valset, batch_size=8, shuffle=False, num_workers=0)

transformer = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.Normalize(mean=0.456, std=0.225)
])

val_loss = 0.0
for i, (inputs, targets) in enumerate(val_loader):
    print(i)
    inputs = inputs.to(device) / 255.0
    targets = targets.to(device) / 255.0
    inputs, targets = transformer.forward(inputs, targets)
    with torch.no_grad():
        outputs = qmodel(inputs)
        loss = utils.kld_loss(outputs, targets)
        val_loss += loss.item()
    if i > 200:
        break

print(val_loss / len(val_loader))

## convert model
qmodel = torch.quantization.convert(qmodel, inplace=False)


torch.save(qmodel.state_dict(), "./checkpoints/salicon-quantized.pt")