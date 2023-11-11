## TensorRT Optimization Example

### Prerequisites

* Install tensorrt python with pip command: `pip install tensorrt`, more details on [NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

### Usage

* Create engine file

```shell
python tensorrt/create_engine.py \
    --weight /path/to/weight_file \
    --input-size 448x448 \
    --input-name "input" \
    --output-name "output" \
    --save_dir "./weights"
```

* Run

```
python tensorrt/inference.py \
    --plan /path/to/engine_file \
    --images /path/to/image_or_folder \
    --output /path/to/output_folder
```