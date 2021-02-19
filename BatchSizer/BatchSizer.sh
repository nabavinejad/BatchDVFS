#!/bin/bash

python InceptionV1_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 131  --result_file '01_InceptionV1_BatchSizer.txt' --topN 1

python InceptionV2_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 123  --result_file '02_InceptionV2_BatchSizer.txt' --topN 1

python InceptionV3_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 228  --result_file '03_InceptionV3_BatchSizer.txt' --topN 1

python InceptionV4_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 84  --result_file '04_InceptionV4_BatchSizer.txt' --topN 1

python MobilenetV1_1_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 145  --result_file '05_MobilenetV1_1_BatchSizer.txt' --topN 1

python MobilenetV1_05_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 89  --result_file '06_MobilenetV1_05_BatchSizer.txt' --topN 1

python MobilenetV1_025_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 112  --result_file '07_MobilenetV1_025_BatchSizer.txt' --topN 1

python MobilenetV2_1_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 89  --result_file '08_MobilenetV2_1_BatchSizer.txt' --topN 1

python MobilenetV2_14_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 164  --result_file '09_MobilenetV2_14_BatchSizer.txt' --topN 1

python NASNet_A_Large_331_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 83  --result_file '10_NASNet_A_Large_331_BatchSizer.txt' --topN 1

python NASNet_A_Mobile_224_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 159  --result_file '11_NASNet_A_Mobile_224_BatchSizer.txt' --topN 1

python PNASNet_5_Mobile_224_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 246  --result_file '12_PNASNet_5_Mobile_224_BatchSizer.txt' --topN 1

python ResNetV2_50_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 143  --result_file '13_ResNetV2_50_BatchSizer.txt' --topN 1

python ResNetV2_101_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 167  --result_file '14_ResNetV2_101_BatchSizer.txt' --topN 1

python ResNetV2_152_BatchSizer.py  --native --batch_size 256  --image_folder 'ILSVRC2012_15000' --power_cap 237  --result_file '15_ResNetV2_152_BatchSizer.txt' --topN 1


python InceptionV1_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 125  --result_file '16_InceptionV1_BatchSizer.txt' --topN 1

python InceptionV2_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 192  --result_file '17_InceptionV2_BatchSizer.txt' --topN 1

python InceptionV3_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 80  --result_file '18_InceptionV3_BatchSizer.txt' --topN 1

python InceptionV4_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 245  --result_file '19_InceptionV4_BatchSizer.txt' --topN 1

python MobilenetV1_1_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 98  --result_file '20_MobilenetV1_1_BatchSizer.txt' --topN 1

python MobilenetV1_05_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 120  --result_file '21_MobilenetV1_05_BatchSizer.txt' --topN 1

python MobilenetV1_025_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 92  --result_file '22_MobilenetV1_025_BatchSizer.txt' --topN 1

python MobilenetV2_1_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 208  --result_file '23_MobilenetV2_1_BatchSizer.txt' --topN 1

python MobilenetV2_14_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 152  --result_file '24_MobilenetV2_14_BatchSizer.txt' --topN 1

python NASNet_A_Mobile_224_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 135  --result_file '25_NASNet_A_Mobile_224_BatchSizer.txt' --topN 1

python PNASNet_5_Large_331_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 86  --result_file '26_PNASNet_5_Large_331_BatchSizer.txt' --topN 1

python PNASNet_5_Mobile_224_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 136  --result_file '27_PNASNet_5_Mobile_224_BatchSizer.txt' --topN 1

python ResNetV2_50_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 198  --result_file '28_ResNetV2_50_BatchSizer.txt' --topN 1

python ResNetV2_101_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 236  --result_file '29_ResNetV2_101_BatchSizer.txt' --topN 1

python ResNetV2_152_BatchSizer.py  --native --batch_size 256  --image_folder 'CalTech_256_20000' --power_cap 79  --result_file '30_ResNetV2_152_BatchSizer.txt' --topN 1


exit 0
