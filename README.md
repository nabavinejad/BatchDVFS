# BatchSizer

BatchSizer is a runtime system that leverages dynamic batching in inference phase of DNNs to control the power consumption of GPU accelerators. As can be seen in the following image, it changes the batch size over the course to manage the power consumption considering power cap.


![BatchSizer](https://github.com/nabavinejad/BatchSizer/blob/main/test.gif)


## Requirements
* TensorFlow GPU (TF V1)
* CUDA
* cuDNN

## DNN Models
We have chosen sixteen DNNs with different characteristics such as size and computational complexity to show the applicability of BatchSizer on a wide variety of DNNs. The DNNs have been selected from [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim). We have followed the instructions provided in the aforementioned library to generate the frozen graphs of the pre-trained models. **You can download the frozen graphs from this [link](TBD)**

## Datasets
We have two image datasets, one from [ImageNet](http://www.image-net.org/) which is a popular dataset that is widely used in other works, and the other one is [CalTech 256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) which is collected by researchers from the California Institute of Technology.

## Usage

There are three folders in the repository: BatchSizer, Clipper, and DVFS that are equivalent to approaches presented in the paper. inside each folder, there is another one that contains the results of the experimented that we have conducted and expalined in the paper. In addition to that, there are 16 .py files that correspond to the 16 DNNs. The .sh file is used for launching the jobs in the experiments. We have assumed that the .py files and the frozen graphs (.pb) are in the same directory. 
