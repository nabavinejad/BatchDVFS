# BatchSizer Approach

In this folder, you can have access to the python files used for conduting the experiments of the BatchSizer approach. The BatchSizer.sh contains all the commands for the experiments of the paper.
For testing new power caps, you can easily use a command as follows:

    python InceptionV1_BatchSizer.py  --native --batch_size 1  --image_folder '/path/to/image/folder' --power_cap desired_power_cap_integer(50 to 250)  --result_file 'Name_Result_File.txt' --topN 1

you can replace InceptionV1_BatchSizer.py with other python files to test other DNNs. Do not forget that we assume the python file and the frozeon graph file are in the same directory.
