# Clipper Approach

This is our implementation of the Clipper approach proposed by Crankshaw et al. [1]. This approach originally uses the
batch size to manage the latency, but we modify
it such that it considers the power cap instead
of latency. Clipper employs an additive-increasemultiplicative-decrease (AIMD) scheme to find the
optimal batch size that maximizes the throughput,
while meeting the latency SLO. We tune Clipper such
that it starts from batch size one and additively increases the batch size by a fixed amount (steps of four
in this work) until the power consumption exceeds
the power cap. At this point, Clipper performs a small multiplicative back-off and reduces the batch
size by 10%. Clipper does not employ DVFS and it
is controlled by internal DVFS controller of the GPU.
Similar to BatchSizer, Clipper starts with batch size of
one.


## Usage

The Clipper.sh contains all the commands for the experiments of the paper. For testing new power caps, you can easily use a command as follows:

    python InceptionV1_Clipper.py  --native --batch_size 1  --image_folder '/path/to/image/folder' --power_cap desired_power_cap_integer(50 to 250)  --result_file 'Name_Result_File.txt' --topN 1

you can replace InceptionV1_Clipper.py with other python files to test other DNNs. Do not forget that we assume the python file and the frozeon graph file are in the same directory.


[1] [Crankshaw, Daniel, Xin Wang, Guilio Zhou, Michael J. Franklin, Joseph E. Gonzalez, and Ion Stoica. "Clipper: A low-latency online prediction serving system." In 14th {USENIX} Symposium on Networked Systems Design and Implementation ({NSDI} 17), pp. 613-627. 2017](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf).
