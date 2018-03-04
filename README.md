# PSPNet_tensorflow
## Introduction
  This is an implementation of PSPNet in TensorFlow for semantic segmentation on the [cityscapes](https://www.cityscapes-dataset.com/) dataset. We first convert weight from [Original Code](https://github.com/hszhao/PSPNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.

## Update:
#### 2018/01/24:
1. `Support evaluation code for ade20k dataset`

#### 2018/01/19:
1. `Support inference phase for ade20k dataset` using model of pspnet50 (convert weights from original author)
2. Using `tf.matmul` to decode label, so as to improve the speed of inference.
#### 2017/11/06:
`Support different input size` by padding input image to (720, 720) if original size is smaller than it, and get result by cropping image in the end.
#### 2017/10/27: 
Change bn layer from `tf.nn.batch_normalization` into `tf.layers.batch_normalization` in order to support training phase. Also update initial model in Google Drive.

## Install 
Get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/1S90PWzXEX_GNzulG1f2eTHvsruITgqsm?usp=sharing) and put into `model` directory. Note: Select the checkpoint corresponding to the dataset.

## Inference
To get result on your own images, use the following command:
```
python inference.py --img-path=./input/test.png --dataset cityscapes  
```
Inference time:  ~0.6s 

Options:
```
--dataset cityscapes or ade20k
--flipped-eval 
--checkpoints /PATH/TO/CHECKPOINT_DIR
```
## Evaluation
### Cityscapes
Perform in single-scaled model on the cityscapes validation datase.

| Method | Accuracy |  
|:-------:|:----------:|
| Without flip| **76.99%** |
| Flip        | **77.23%** |

### ade20k
| Method | Accuracy |  
|:-------:|:----------:|
| Without flip| **40.00%** |
| Flip        | **40.67%** |

To re-produce evluation results, do following steps:
1. Download [Cityscape dataset](https://www.cityscapes-dataset.com/) or [ADE20k dataset](http://sceneparsing.csail.mit.edu/) first. 
2. change `data_dir` to your dataset path in `evaluate.py`:
```
'data_dir': ' = /Path/to/dataset'
```
3. Run the following command: 
```
python evaluate.py --dataset cityscapes
```
List of Args:
```
--dataset - ade20k or cityscapes
--flipped-eval  - Using flipped evaluation method
--measure-time  - Calculate inference time
```

## Image Result
### cityscapes
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/test1.png)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/test_1024x2048.png)
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/test_720x720.png)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/test_720x720.png)

### ade20k
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/indoor_2.jpg)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/indoor_2.jpg)

### real world
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/indoor_1.jpg)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/indoor_1.jpg)

### Citation
    @article{zhao2017pspnet,
      author = {Hengshuang Zhao and
                Jianping Shi and
                Xiaojuan Qi and
                Xiaogang Wang and
                Jiaya Jia},
      title = {Pyramid Scene Parsing Network},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
    }
Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. arXiv:1608.05442. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2016semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={arXiv preprint arXiv:1608.05442},
      year={2016}
    }
