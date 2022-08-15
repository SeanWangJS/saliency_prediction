import os
import json
import argparse
import glob
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage
import cv2
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def parse_annotation(annotation_file: str):
    """
    Parse fixation_train2014.json | fixation_val2014.json file
    :params: annotation_file: File of annotation
    """
    with open(os.path.join(annotation_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    ## create dict: {"image_id": image_info}
    image_info_dict = {}
    for image in data["images"]:
        image_id=image["id"]
        image_info_dict[image_id] = image

    ## create dict: {"image_id": annotations}
    annotations=data["annotations"]
    image_annotations_dict = {}
    for ann in annotations:
        fixations=ann["fixations"]
        image_id=ann["image_id"]
        if image_id in image_annotations_dict:
            image_annotations_dict[image_id].append(fixations)
        else:
            image_annotations_dict[image_id]= [fixations]
        
    return image_info_dict, image_annotations_dict

def sigmoid(x):
    return 1 / (1 + np.exp(- 8*(x-0.7)))

def generate_saliency_map(fixations: list, img_width: int, img_height: int, save_file: str):
    """
    Create saliency map of image with fixations
    :params: fixations: list of fixations
    :params: img_width: width of image
    :params: img_height: height of image
    """
    img = np.zeros((img_height, img_width))
    for p in fixations:
        img[p[0]-1, p[1]-1] = 1

    blur=ndimage.filters.gaussian_filter(img, 19)
    blur -= blur.min()
    blur /= blur.max()
    blur=sigmoid(blur) * 255
    blur = blur.astype(np.uint8)
    cv2.imwrite(save_file, blur)

def generate_sliency_maps(salicon_dir: str, annotation_file: str, executor: ThreadPoolExecutor):
    logging.info("Reading annotation file: {}".format(annotation_file))

    s = time.time()

    split_name = os.path.basename(annotation_file)
    split_name = os.path.splitext(split_name)[0]

    output_dir = os.path.join(salicon_dir, f"{split_name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_info_dict, image_annotation_dict = parse_annotation(annotation_file)

    e = time.time()
    logging.info("Time to parse annotation file: {}s".format(e-s))

    items = list(image_info_dict.items())

    def exec(i: int):

        image_id, image_info = items[i]
        img_width = image_info["width"]
        img_height = image_info["height"]
        file_name = image_info["file_name"]
        annotation = image_annotation_dict[image_id]
        fixations = [item for sublist in annotation for item in sublist]
        save_path = f"{output_dir}/{file_name}"

        img = np.zeros((img_height, img_width))
        
        for p in fixations:
            img[p[0]-1, p[1]-1] = 1

        blur=ndimage.filters.gaussian_filter(img, 19)
        blur -= blur.min()
        blur /= blur.max()
        blur=sigmoid(blur) * 255
        blur = blur.astype(np.uint8)
        cv2.imwrite(save_path, blur)

    list(tqdm(executor.map(exec, range(len(items))), total=len(items)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--salicon_dir", type=str, help="Directory of SALICON dataset")
    parser.add_argument("--num_threads", type=int, help="Number of threads", default=1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    salicon_dir =args.salicon_dir
    num_threads = args.num_threads

    executor = ThreadPoolExecutor(max_workers=num_threads)

    annotation_fils = glob.glob(salicon_dir + "/*.json")
    logging.info("Start generating saliency maps")

    for annotation_file in annotation_fils:
        generate_sliency_maps(salicon_dir, annotation_file, executor)
    
