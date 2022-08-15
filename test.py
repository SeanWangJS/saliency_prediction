import PIL.Image as Image
import glob
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import cv2
import numpy as np

from models.salicon import SaliconNet, QuantizableSaliconNet
import torch.nn.functional as F

import argparse
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", dest="images", default="imgs", help = "Image or Directory containing images")
    parser.add_argument("--checkpoint", help="Checkpoint file")
    parser.add_argument("--output", default="output", dest="output", help="Output folder to save result images")
    parser.add_argument("--quant", default=False, dest="quant", help="Use quantization model")
    parser.add_argument("--bounding-box", default=False, dest="bounding_box", help="Draw bounding box on saliency map")
    
    return parser.parse_args()

def logistic(x):
    return 1 / (1 + torch.exp(-10 * (x-0.7)))

def softmax(x: torch.Tensor):
    exp_x = x.exp()
    sum_exp_x = exp_x.sum(dim=[2,3], keepdim=True)
    return exp_x / sum_exp_x

def load_model(use_quant: bool, checkpoint: str):
    if use_quant:
        model = QuantizableSaliconNet()
        device = 'cpu'
        model.to(device)
        model.eval()
        model.fuse_model()
        ## setting qconfig
        backend = "fbgemm"
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        ## prepare
        qmodel = torch.quantization.prepare(model, inplace=False)
        ## convert model
        qmodel = torch.quantization.convert(qmodel, inplace=False)
        qmodel.load_state_dict(torch.load(checkpoint))
        model = qmodel
    else:
        model = SaliconNet()
        model.load_state_dict(torch.load(checkpoint))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval()
    
    return model, device

if __name__ == '__main__':

    args = arg_parse()

    images = args.images
    if os.path.isdir(images):
        images = glob.glob(images + "/*")
    else:
        images = [images]

    transformer = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.456, std=0.225)
    ])

    model, device = load_model(args.quant, args.checkpoint)
    
    for path in images:
        img=Image.open(path).convert('RGB')
        w = img.width
        h = img.height
        x = transformer(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x)

        out = y
        out -= out.min()
        out /= out.max()

        # out = logistic(out) * 255
        out = out * 255
                
        img=out[0,0].detach().cpu().numpy().astype(np.uint8)

        img=cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # img=cv2.GaussianBlur(img, (11, 11), sigmaX=2, sigmaY = 2)
        filename=os.path.basename(path)

        cv2.imwrite(args.output + "/" + filename, img)
