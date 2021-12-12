# BOBD
The BatchDVFS approach is a lightweight runtime system that can adjust the batch size and DVFS in an online manner, and hence, 
is appropriate for making immediate decisions to avoid power cap violation when the job starts execution, 
or when the power cap is changed during execution of job. However, this approach cannot guarantee to find the optimal or near-optimal solution 
that can maximize the throughput while meeting the power cap, as expected. 
It can render the performance of job low, especially when the power cap is constants for a long period of time and job is long-running.

To address this challenge, we design and implement an offline approach leveraging Bayesian Optimization (BO) 
that has been employed in other works as well \cite{patel2020clite,li2021rambo,alipourfard2017cherrypick, nabavinejad2021bayestuner}. This approach, which we call \textit{Bayesian Optimization for coordinating BS and DVFS (BOBD)}, can find the combination of batch size and DVFS that leads to optimal or near-optimal solution, but with a significant time overhead. Hence, it cannot be used instead of \textit{BatchDVFS}, but complements it. While overhead of this approach is significant compared to \textit{BatchDVFS}, it still imposes much lower overhead compared to an exhaustive approach that aims to find the optimal solution by testing all the possible combinations of batch size and DVFS.

## Requirements
* TensorFlow GPU (TF V1)
* CUDA
* cuDNN

## DNN Models
We have chosen sixteen DNNs with different characteristics such as size and computational complexity to show the applicability of BatchDVFS on a wide variety of DNNs. The DNNs have been selected from [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim). We have followed the instructions provided in the aforementioned library to generate the frozen graphs of the pre-trained models. **You can download the frozen graphs from this [link](https://drive.google.com/file/d/1QJFxeoO_gmZiK-vzM75OQnA0XjL5ZL9P/view?usp=sharing)**

## Datasets
We have two image datasets, one from [ImageNet](http://www.image-net.org/) which is a popular dataset that is widely used in other works, and the other one is [CalTech 256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) which is collected by researchers from the California Institute of Technology.

## Usage

There are three folders in the repository: BatchDVFS, Clipper, and DVFS that are equivalent to approaches presented in the paper. Inside each folder, there is another one that contains the results of the experiments that we have conducted and presented in the paper. In addition to that, there are 16 .py files that correspond to the 16 DNNs. The .sh file is used for launching the jobs in the experiments. We have assumed that the .py files and the frozen graphs (.pb) are in the same directory.

Moreover, there is an extra folder in BatchDVFS folder, named SensitiviyAnalysis that includes the files needed for conducting the sensitivity analysis experiments. The .py files in this folder are slightly different from the ones in the main folder and enable the dynmiac power cap change during the execution of a model.
