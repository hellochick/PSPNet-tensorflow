# PSPNet_tensorflow
## Introduction
  This is an implementation of PSPNet in TensorFlow for semantic segmentation on the [cityscapes](https://www.cityscapes-dataset.com/) dataset. We first convert weight from [Original Code](https://github.com/hszhao/PSPNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.

## Install
First get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/0B9CKOTmy0DyacmNpNlhyb1lmbU0?usp=sharing) and put into `model` directory.

## Inference
To get result on your own images, use the following command:
```
python inference.py --img-path=./input/test.png
```

## Evaluation
Perform in single-scaled model `without flipped evaluation and sliding-window method` mentioned in original paper, we gets `76.99% mIoU` on the cityscapes validation datase.

To get evaluation result, use the following command: 
```
python evaluate.py
```

## Image Result
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/test.png)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/test.png)
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/test2.png)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/test2.png)
