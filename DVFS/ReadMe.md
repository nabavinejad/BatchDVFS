# DVFS Approach

Dynamic Voltage Frequency Scaling (DVFS) approach is another rival that we have compared our BatchSizer against. In this approach (DVFS), the batch size is constant and only the 
frequency of GPU is dynamically adjust to manage the power consumption. Two version of this approach is considered: DVFS_1 
where the batch size is equal to one and DVFS_256 where the batch size is equal to 256. 


In this folder, you can have access to the python files used for conduting the experiments of the DVFS approach. 
The DVFS_1.sh and DVFS_256.sh contain all the commands for the experiments of the paper. The resuls we have used in our paper is accessible via DVFS_1 and DVFS_256 folders.
For testing new power caps, you can easily use a command as follows:

    python InceptionV1_DVFS.py  --native --batch_size (1, 256, or any other value)  --image_folder '/path/to/image/folder' --power_cap desired_power_cap_integer(50 to 250)  --result_file 'Name_Result_File.txt' --topN 1

you can replace InceptionV1_DVFS.py with other python files to test other DNNs. Do not forget 
that we assume the python file and the frozeon graph file are in the same directory.
