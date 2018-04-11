# CNTK Examples: Image/Detection/Faster R-CNN

## Overview

This folder contains an end-to-end solution for using Faster R-CNN to perform object detection. 
The original research paper for Faster R-CNN can be found at [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497).
Base models that are supported by the current configuration are AlexNet and VGG16. 
Two image set that are preconfigured are Pascal VOC 2007 and Grocery. 
Other base models or image sets can be used by adapting config.py.

## Running the code

### Setup

To run Faster R-CNN you need a CNTK Python environment. Install additional packages:

```
pip install -r requirements.txt
```

The code uses prebuild Cython modules for parts of the region proposal network (see `Examples/Image/Detection/utils/cython_modules`). 
These binaries are contained in the repository for Python 3.5 under Windows and Python 3.4 under Linux.
If you require other versions please follow the instructions at [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#installation-sufficient-for-the-demo).

### Getting the data and AlexNet model

We use a toy dataset of images captured from a refrigerator to demonstrate Faster R-CNN. Both the dataset and the pre-trained AlexNet model can be downloaded by running the following Python command:

`python install_data-and-model.py`

After running the script, the toy dataset will be installed under the `Image/DataSets/Grocery` folder. And the AlexNet model will be downloaded to the `Image/PretrainedModels` folder. 
We recommend you to keep the downloaded data in the respective folder while downloading, as the configuration files in this folder assumes that by default.

## Technical details

### Parameters

All options and parameters are in `config.py`. These include

```
__C.CNTK.DATASET = "Grocery"    # "Grocery" or "Pascal"
__C.CNTK.BASE_MODEL = "AlexNet" # "VGG16" or "AlexNet"

__C.CNTK.TRAIN_E2E = True       # E2E or 4-stage training

__C.CNTK.E2E_MAX_EPOCHS = 20
__C.CNTK.E2E_LR_PER_SAMPLE = [0.001] * 10 + [0.0001] * 10 + [0.00001]
```

However, instead of modifying this Python file, simply supply a YAML config
to the `FasterRCNN.py` script with all the parameters using the `--config` option.

### Algorithm 

All details can be found in the original research paper: [https://arxiv.org/abs/1506.01497](https://arxiv.org/abs/1506.01497).

