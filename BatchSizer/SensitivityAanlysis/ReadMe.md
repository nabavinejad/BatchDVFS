# BatchSizer Sensitivity Analysis
To conduct the sensitivity analysis experiments, we use the files in this folder. The results are also presented here. For running the experiments, we can proceed as follows (similar to other scripts):

    python InceptionV4_BatchSizer_Sensitivity.py  --native --batch_size 256  --image_folder '/path/to/image/folder' --power_cap 150  --result_file 'InceptionV4_BatchSizer_Sensitivity.txt' --topN 1
    
    python ResNetV2_152_BatchSizer_Sensitivity.py  --native --batch_size 256  --image_folder '/path/to/image/folder' --power_cap 150  --result_file 'ResNetV2_152_BatchSizer_Sensitivity.txt' --topN 1
    
    python PNASNet_5_Large_331_BatchSizer_Sensitivity.py  --native --batch_size 256  --image_folder '/path/to/image/folder' --power_cap 100  --result_file 'PNASNet_5_Large_331_BatchSizer_Sensitivity.txt' --topN 1
    
    python MobilenetV1_1_BatchSizer_Sensitivity.py  --native --batch_size 256  --image_folder '/path/to/image/folder' --power_cap 75  --result_file 'MobilenetV1_1_BatchSizer_Sensitivity.txt' --topN 1

Here we again assume that the python files (e.g, InceptionV4_BatchSizer_Sensitivity.py) and the forzeon graphs (e.g., frozen_inception_v4.pb) are in the same directory.
