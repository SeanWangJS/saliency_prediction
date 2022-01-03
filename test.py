import PIL.Image as Image
import glob
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np

from models.salicon import SaliconNet
import torch.nn.functional as F

import argparse
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", dest="images", default="imgs", help = "Image or Directory containing images")
    parser.add_argument("--checkpoint", help="Checkpoint file")
    parser.add_argument("--output", default="output", dest="output", help="Output folder to save result images")
    
    return parser.parse_args()

def logistic(x):
    return 1 / (1 + torch.exp(-10 * (x-0.7)))

if __name__ == '__main__':
    args = arg_parse()

    images = args.images
    if os.path.isdir(images):
        images = glob.glob(images + "/*.jpg")
    else:
        images = [images]

    transformer = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.456, std=0.225)
    ])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model = SaliconNet()
    
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    
    for path in images:
        img=Image.open(path).convert('RGB')
        w = img.width
        h = img.height
        x = transformer(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x)

        out=y[0,0]

        w_out, h_out = out.shape
        out = out.reshape(w_out * h_out, -1)
        out = F.softmax(out, dim=0)
        out = out.reshape(w_out, h_out)


        mx = out.max()
        mi = out.min()

        out=(out - mi) / (mx - mi)
        # out = out * 255
        out = logistic(out) * 255
        img=out.detach().cpu().numpy().astype(np.uint8)
        img=cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        img=cv2.GaussianBlur(img, (11, 11), sigmaX=2, sigmaY = 2)

        filename=os.path.basename(path)
        cv2.imwrite(args.output + "/" + filename, img)
