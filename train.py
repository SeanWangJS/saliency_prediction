import argparse

from unet import UNet
from models.res_unet import ResUNet
from models.salicon import SaliconNet
from dataset import SaliconDataset

from typing import List

import torch
from torch.utils.data.dataloader import DataLoader

import transforms
import utils
import time

from tensorboardX import SummaryWriter

import os

writer = SummaryWriter()
device = "cuda" if  torch.cuda.is_available()  else "cpu"

torch.set_num_threads(1)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", help="Directory of Data", default="/opt/dataset/SALICON")
    parser.add_argument("--arch", dest="arch", help="Network architecture", default="salicon")
    parser.add_argument("--batch_size", dest="batch_size", default=16, help="Batch size for training")
    parser.add_argument("--nepochs", type=int, dest="nepochs", default=60, help="number of epochs")
    parser.add_argument("--checkpoints_dir", dest="checkpoints_dir", default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--recover", dest="recover", help="Continue traing with exist checkpoint file, ignore if not specify checkpoint file", default=False)
    parser.add_argument("--checkpoint_file", dest="checkpoint_file")
    parser.add_argument("--start_epoch", dest="start_epoch")

    return parser.parse_args()

def train_step(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor, optimizer: torch.optim.Optimizer, criterion):

    outputs = model(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(epoch: int, model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion, transformer: torch.nn.Module):
    model.train()
    train_loss = 0
    init_time = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device) / 255.0
        targets = targets.to(device) / 255.0
        inputs, targets = transformer.forward(inputs, targets)
        loss = train_step(model, inputs, targets, optimizer, criterion)
        train_loss += loss
        if i % 10 == 0:
            e = time.time()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {}'.format(
                epoch, i * len(inputs), len(train_loader.dataset),
                100. * i / len(train_loader), loss, e - init_time))
            init_time = e
    writer.add_scalar('train_loss', train_loss / len(train_loader), epoch)

def val_epoch(epoch: int, model: torch.nn.Module, val_loader: DataLoader, criterion, transformer: torch.nn.Module):
    model.eval()
    val_loss = 0
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device) / 255.0
        targets = targets.to(device) / 255.0
        inputs, targets = transformer.forward(inputs, targets)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss
    writer.add_scalar('val_loss', val_loss / len(val_loader), epoch)

if __name__ == '__main__':
    args = arg_parse()

    data_dir = args.data_dir
    arch = args.arch
    checkpoints_dir = args.checkpoints_dir
    batch_size=int(args.batch_size)
    nEpochs = args.nepochs

    if arch == "unet":
        model = UNet().to(device)
    elif arch == "res_unet":
        model = ResUNet().to(device)
    elif arch == "salicon":
        model = SaliconNet()
    else:
        raise Exception("Unknown network architecture: " + arch)
    
    start_epoch = 0
    if args.recover and os.path.exists(args.checkpoint_file):
        start_epoch = int(args.start_epoch)
        model.load_state_dict(torch.load(args.checkpoint_file))

    model.to(device)

    transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(600),
        transforms.Normalize(mean=0.456, std=0.225)
    ])

    trainset=SaliconDataset(data_dir, "train", (800, 600))
    valset = SaliconDataset(data_dir, "val", (800, 600))
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=2, shuffle=False, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, num_workers=2, shuffle=False)

    optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.01)
    criterion = utils.kld_loss

    for epoch in range(start_epoch, start_epoch + nEpochs, 1):
        train_epoch(epoch, model, train_loader, optimizer, criterion, transformer)
        scheduler.step()
        val_epoch(epoch, model, val_loader, criterion, transformer)
        torch.save(model.state_dict(), f"{checkpoints_dir}/salicon-{arch}-new2-{epoch}.pth")