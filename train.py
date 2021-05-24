import argparse

from unet import UNet
from res_unet import ResUNet
from dataset import SaliconDataset

from typing import List

import torch
from torch.utils.data.dataloader import DataLoader

import transforms

from tensorboardX import SummaryWriter

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", dest="images_dir", help="Directory of Images")
    parser.add_argument("--fixations_dir", dest="fixations_dir", help="Directory of Fixation Images")
    parser.add_argument("--train_csv", dest="train_csv", help="CSV file contains train images name")
    parser.add_argument("--arch", dest="arch", help="Network architecture")
    parser.add_argument("--batch_size", dest="batch_size", default=2, help="Batch size for training")
    parser.add_argument("--nepochs", type=int, dest="nepochs", default=10, help="number of epochs")
    parser.add_argument("--checkpoints_dir", dest="checkpoints_dir", default="checkpoints", help="Where to save checkpoints")

    return parser.parse_args()

def loss_func(outputs: List[torch.Tensor], labels: List[torch.Tensor]):
    loss = 0
    for output, label in zip(outputs, labels):
        l = (output / output.max() - label) / (alpha - label)
        l = (l**2).sum()
        loss += l
    return loss / len(outputs) /  2500


if __name__ == '__main__':
    args = arg_parse()

    images_dir = args.images_dir
    fixations_dir = args.fixations_dir
    train_csv = args.train_csv
    arch = args.arch
    checkpoints_dir = args.checkpoints_dir
    batch_size=int(args.batch_size)
    nEpochs = args.nepochs

    writer=SummaryWriter()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if arch == "unet":
        net = UNet().to(device)
    elif arch == "res_unet":
        net = ResUNet().to(device)
    else:
        raise Exception("Unknown network architecture: " + arch)

    transformer = transforms.Compose([
        transforms.Resize([556, 556]),
        transforms.RandomCrop([512, 512]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    trainset=SaliconDataset(images_dir, fixations_dir, train_csv, transformer)
    
    train_lodaer = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=False)

    optimizer=torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    alpha=1.1

    criterion=loss_func

    step=0
    net.train()
    
    for epoch in range(nEpochs):
        for i, (inputs, labels) in enumerate(train_lodaer):
            
            inputs=inputs.to(device)
            labels = labels.to(device)
            N, C, H, W = labels.shape
            labels=labels.reshape(N, H, W)

            outputs=net(inputs)
            outputs=outputs.reshape(N, H, W)

            optimizer.zero_grad()

            outputs = [output for output in outputs]
            labels = [label for label in labels]

            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch}/{nEpochs} | Step: {i}/{len(trainset) // batch_size} | EMD Loss: {loss.item()}')
            step = step + 1

            writer.add_scalar('batch train loss', loss.item(), i + epoch * (len(trainset) // batch_size + 1))
        
        torch.save(net.state_dict(), f"{checkpoints_dir}/salicon-{arch}-{epoch}.pth")
