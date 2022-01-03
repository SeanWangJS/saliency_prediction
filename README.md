## SALICON

![](./resources/index.png)

Customized implementation of the paper [SALICON: Reducing the Semantic Gap in Saliency Prediction by Adapting Deep Neural Networks](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Huang_SALICON_Reducing_the_ICCV_2015_paper.pdf) by PyTorch for saliency prediction. This model was trained from scratch with Salicon Dataset which is available on http://salicon.net/download/.

### Usage

#### Dataset Preparing

Download Salicon dataset and then generate the saliency map by using fixation_train/val/test2014.json files. See [utils.py](utils.py) for detail, and the dataset examples are shown in directory ./data_sample

#### Prediction

Get the checkpoint file from [here](https://drive.google.com/file/d/15wD0XMI6NPWDeJX2n7R6gAerZcHY3XWM/view?usp=sharing).

```shell
python test.py --images=/path/to/image_or_folder --checkpoint=/path/to/your/checkpoint_file --output=/folder/to/output_images
```

#### Train

```shell
python train.py --data_dir=/path/to/data_folder
```

### Result

Saliency prediction examples are shown below

![](./resources/result.jpg)