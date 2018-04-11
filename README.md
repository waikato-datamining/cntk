# CNTK

Collection of modified CNTK scripts. Usually based on the example
scripts, but modified to be useful in production environments.

See version directories for installation instructions and further
information.

## Pretrained models

### CNTK

Native CNTK models can be downloaded from URLs found in the 
following Python module: 

https://github.com/Microsoft/CNTK/blob/master/PretrainedModels/download_model.py

Examples:

* [Inception v3](https://www.cntk.ai/Models/CNTK_Pretrained/InceptionV3_ImageNet_CNTK.model)
* [ResNet 18](https://www.cntk.ai/Models/CNTK_Pretrained/ResNet18_ImageNet_CNTK.model)
* [VGG 16](https://www.cntk.ai/Models/Caffe_Converted/VGG16_ImageNet_Caffe.model)


### ONNX

CNTK can load models in ONNX format since 2.3.

* ONNX models: 

  https://github.com/onnx/models
  
* how to load an ONNX model with Python 
  ([source](https://docs.microsoft.com/en-us/cognitive-toolkit/serialization)):

  ```python
  import cntk as C
  z = C.Function.load("myModel.onnx", format=C.ModelFormat.ONNX)
  ```
  
* how to save a model in ONNX format with Python:

  ```python
  import cntk as C
  z = ...
  z.save("myModel.onnx", format=C.ModelFormat.ONNX)
  ```
