import os
import sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
import argparse

import tensorrt as trt
import torch

from models.salicon import SaliconNet

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", dest = "weight", help = "Path to the weight file", required=True)
    parser.add_argument("--input-size", dest = "input_size", help = "Input size: wxh", default="448x448")
    parser.add_argument("--input-name", dest = "input_name", help = "Input tensor name", default="input")
    parser.add_argument("--output-name", dest = "output_name", help = "Output tensor name", default="output")
    parser.add_argument("--save_dir", help = "Directory to save engine plan", default="./weights")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    weight_path = args.weight
    input_size = args.input_size
    input_name = args.input_name
    output_name = args.output_name
    save_dir_path = args.save_dir

    try:
        w, h = input_size.split("x")
        w = int(w)
        h = int(h)
    except:
        print("Invalid input size, try `wxh` format")
        exit(1)

    model = SaliconNet()
    model.load_state_dict(torch.load(weight_path))

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    ## export onnx
    dummy_input = torch.randn(1, 3, h, w)
    onnx_path = os.path.join(save_dir_path, "model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, input_names=[input_name], output_names=[output_name])

    print("Exported onnx model to {}".format(onnx_path))

    logger  = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    engineString = builder.build_serialized_network(network, config)

    engine_path = os.path.join(save_dir_path, "model.plan")
    with open(engine_path, "wb") as f:
        f.write(engineString)
    
    print("Serialize engine plan to {}".format(engine_path))


