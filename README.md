# PSPNet_tensorflow
## Introduction
  This is an implementation of PSPNet in TensorFlow for semantic segmentation on the [cityscapes](https://www.cityscapes-dataset.com/) dataset. We first convert weight from [Original Code](https://github.com/hszhao/PSPNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.

## Install
Get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/0B9CKOTmy0DyacmNpNlhyb1lmbU0?usp=sharing) and put into `model` directory.

## Inference
To get result on your own images, use the following command:
```
python inference.py --img-path=./input/test.png
```
Inference time:  ~0.6s 

## Evaluation
Perform in single-scaled model on the cityscapes validation datase.

| Method | Accuracy |  
|:-------:|:----------:|
| Without flip| **76.99%** |
| Flip        | **77.23%** |

To get evaluation result, you need to download Cityscape dataset from [Official website](https://www.cityscapes-dataset.com/) first. Then change `DATA_DIRECTORY` to your dataset path in `evaluate.py`:
```
DATA_DIRECTORY = /Path/to/dataset
```

Then run the following command: 
```
python evaluate.py
```
List of Args:
```
--flipped-eval  - Using flipped evaluation method
--measure-time  - Calculate inference time
```
## Image Result
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/test.png)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/test.png)
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/test2.png)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/test2.png)
