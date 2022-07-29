import numpy as np
import pandas as pd
import cv2
import json

from scipy import ndimage
import torch
import torch.nn.functional as F

def kld_loss(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Calculates the kld loss
    """
    b, c, h, w = outputs.shape

    outputs = outputs.reshape(b, c, -1)
    outputs=F.softmax(outputs, dim=2)

    print(outputs.log())

    labels=F.interpolate(labels, size=(h, w), mode="bilinear", align_corners=False)
    labels = labels.reshape(b, c, -1)
    labels = F.softmax(labels, dim=2)

    print(labels)

    return F.kl_div(outputs, labels, reduction='batchmean')

def bce(outputs: torch.Tensor, labels: torch.Tensor):
    b, c, h, w = outputs.shape

    labels=F.interpolate(labels, size=(h, w), mode="bilinear", align_corners=False)
    labels = labels / 255.0
    # print(labels)
    # print(outputs)

    loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
    return loss
    

def bce_loss(outputs: torch.Tensor, labels: torch.Tensor):
    """
    Calculates the binary cross entroy loss
    """
    return -torch.sum(labels * torch.log(outputs + 1e-7) + (1 - labels) * torch.log(1 - outputs + 1e-7))

def loss_func(outputs: torch.Tensor, labels: torch.Tensor):
    loss = 0
    alpha = 1.1
    for output, label in zip(outputs, labels):
        l = (output / output.max() - label) / (alpha - label)
        l = (l**2).sum()
        loss += l
    return loss / len(outputs) /  2500

def sigmoid(x):
    return 1 / (1 + np.exp(- 8*(x-0.7)))

def generate(data: dict, split: str, result_dir: str="./data"):
    """
        Generate saliency map
        Args:
            data: fixations json file in SALICON dataset, could be one of fixation_train2014.json or fixation_val2014.json or fixation_test2014.json
            split: One of "train", "val", "test"
            result_dir: Directory where to save the saliency maps
    """

    ## create dict: {"image_id": image_json}
    image_dicts={}
    for image in data["images"]:
        image_id=image["id"]
        image_dicts[image_id] = image

    ## create dict: {"image_id": fixations}
    annotations=data["annotations"]
    image_anns = {}
    for ann in annotations:
        fixations=ann["fixations"]
        image_id=ann["image_id"]
        if image_id in image_anns:
            image_anns[image_id].append(fixations)
        else:
            image_anns[image_id]= [fixations]
    
    k = 0
    
    image_names=[]
    for image_id in image_dicts.keys():
        print(k)
        if k>100:
            break
        k = k + 1

        image_json = image_dicts[image_id]
        file_name = image_json["file_name"]
        image_names.append(file_name)
        anns = image_anns[image_id]

        fixations = [item for sublist in anns for item in sublist]
        w = image["width"]
        h = image["height"]
        img = np.zeros((h, w))
        for p in fixations:
            img[p[0]-1, p[1]-1] = 1

        blur=ndimage.filters.gaussian_filter(img, 19)
        # mx = blur.max()
        # mi = blur.min()
        # blur=(blur - mi) / (mx - mi)
        blur -= np.min(blur)
        blur /= np.max(blur)
        blur=sigmoid(blur) * 255
        # blur = blur * 255
        blur = blur.astype(np.uint8)

        cv2.imwrite(f"{result_dir}/fixations/" + file_name, blur)
    image_names=pd.Series(image_names)
    image_names.to_csv(f"{result_dir}/{split}-2.csv", header=False, index=False)

# # path = "D:/data/Datasets/SALICON/fixations_train2014.json"
# path = "./annotations/fixations_train2014examples.json"
# with open(path, "r") as f:
#     data=json.load(f)

# # output_path="D:/data/Datasets/SALICON/train_fixations2"
# output_path="./data"
# generate(data, "train", output_path)