import PIL.Image as Image
import glob
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np

from res_unet import ResUNet
from unet import UNet

import argparse
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", dest="images", default="imgs", help = "Image or Directory containing images")
    parser.add_argument("--checkpoint", help="Checkpoint file")
    parser.add_argument("--arch", dest="arch", default="unet", help="Specify the model architecture")
    parser.add_argument("--output", default="output", dest="output", help="Output folder to save result images")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()

    images = args.images
    if os.path.isdir(images):
        images = glob.glob(images + "/*.jpg")
    else:
        images = [images]

    transformer = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    arch = args.arch
    if arch == "unet":
        net = UNet().to(device)
    elif arch == "res_unet":
        net = ResUNet().to(device)
    
    net.load_state_dict(torch.load(args.checkpoint))
    
    for path in images:
        print(path)
        img=Image.open(path)
        w = img.width
        h = img.height
        x = transformer(img).unsqueeze(0).to(device)
        print(x.shape)
        y = net(x)

        out=y[0,0]
        mx = out.max()
        mi = out.min()
        out=(out - mi) / (mx - mi) * 255
        img=out.detach().cpu().numpy().astype(np.uint8)
        img=cv2.resize(img, (w, h))

        filename=os.path.basename(path)
        cv2.imwrite(args.output + "/" + filename, img)


