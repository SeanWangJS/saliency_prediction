import argparse
import os
import glob
import time

import tensorrt as trt
from cuda import cudart
import numpy as np
import cv2

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", dest="plan", help="Path to the engine plan file", required=True)
    parser.add_argument("--images", dest="images", help="Path to the image file/image folder", required=True)
    parser.add_argument("--output", dest="output", help="Path to the output folder", default="./output")

    return parser.parse_args()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class run_time:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"{self.name} time: {elapsed_time * 1000}ms")

def preprocess(img: np.ndarray, input_size: tuple):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))

    for i in range(3):
        img[i] = (img[i] - MEAN[i]) / STD[i]

    img = img[np.newaxis, ...]
    return img

def postprocess(output: np.ndarray, orig_size: tuple):

    output -= output.min()
    output /= output.max()
    output *= 255.0
    img = output.astype(np.uint8)
    img = cv2.resize(img, orig_size)
    return img

if __name__ == "__main__":

    args = arg_parse()

    plan_path = args.plan
    image_path = args.images
    output_path = args.output

    if os.path.isdir(image_path):
        image_paths = glob.glob("{}/*".format(image_path))
    else:
        image_paths = [image_path]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(plan_path, "rb") as f:
        engineString = f.read()

    logger  = trt.Logger(trt.Logger.VERBOSE)
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()

    input_name  = engine.get_tensor_name(0)
    input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(input_name)))
    input_shape = context.get_tensor_shape(input_name)
    input_size  = np.prod(input_shape) * input_dtype.itemsize

    output_name  = engine.get_tensor_name(1)
    output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output_name)))
    output_shape = context.get_tensor_shape(output_name)
    output_size  = np.prod(output_shape) * output_dtype.itemsize

    ## allocate cuda memory
    input_ptr_d = cudart.cudaMalloc(input_size)[1]
    output_ptr_d = cudart.cudaMalloc(output_size)[1]

    for img_path in image_paths:
        filename = os.path.basename(img_path)

        img = cv2.imread(img_path)
        with run_time("Preprocess"):
            input = preprocess(img, (input_shape[2], input_shape[3]))
            input = np.ascontiguousarray(input)
        input_ptr_h = input.ctypes.data

        ## copy input to device
        cudart.cudaMemcpy(input_ptr_d, input_ptr_h, input_size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        ## execute
        context.set_tensor_address(input_name, input_ptr_d)
        context.set_tensor_address(output_name, output_ptr_d)
        with run_time("Inference"):
            context.execute_async_v3(0)
            cudart.cudaStreamSynchronize(0)

        ## copy output to host
        output = np.empty(output_shape, dtype=output_dtype)
        output_ptr_h = output.ctypes.data
        cudart.cudaMemcpy(output_ptr_h, output_ptr_d, output_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        img_size = [img.shape[1], img.shape[0]]
        output_img = postprocess(output[0, 0], img_size)
        
        cv2.imwrite(os.path.join(output_path, filename), output_img)
    
    cudart.cudaFree(input_ptr_d)
    cudart.cudaFree(output_ptr_d)



