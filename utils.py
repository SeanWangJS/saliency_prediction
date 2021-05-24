import numpy as np
import pandas as pd
import cv2

from scipy import ndimage

def sigmoid(x):
    return 1 / (1 + np.exp(- 10*(x-0.5)))


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
        mx = blur.max()
        mi = blur.min()
        blur=(blur - mi) / (mx - mi)
        blur=sigmoid(blur) * 255
        blur = blur.astype(np.uint8)

        cv2.imwrite(f"{result_dir}/fixations/" + file_name, blur)
    image_names=pd.Series(image_names)
    image_names.to_csv(f"{result_dir}/{split}.csv", header=False, index=False)