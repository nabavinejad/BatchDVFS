# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/bin/env python -tt
r""" TF-TensorRT integration sample script """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import tensorflow.contrib.tensorrt as trt
from subprocess import call

import numpy as np
import time
import math
import os
import glob
from tensorflow.python.platform import gfile
from tensorflow.python.client import timeline
import argparse, sys, itertools,datetime
import json
tf.logging.set_verbosity(tf.logging.INFO)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device


def _parse_function(filename):
    #tic = time.time()
    input_height=224
    input_width=224
    input_mean=0
    input_std=255
    """ Read a jpg image file and return a tensor """
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(filename, input_name)
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3)
    float_caster = tf.cast(image_reader, tf.float32)/128. - 1
    #resized = tf.image.resize_images(float_caster, [input_height, input_width])
    resized = tf.image.resize_images(float_caster, [input_height, input_width])
    ##  image_batch = tf.train.batch([resized], batch_size=4)
    #print(time.time() - tic)
    return resized


def getResnet50():
  with gfile.FastGFile("frozen_resnet_v2_101.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def timeGraph(gdef,batch_size=128,image_folder='images',nvidiasmi='output.out', latencyF = 'latency.txt', StopTime = 100):
  tf.logging.info("Starting execution")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
  tf.reset_default_graph()
  g = tf.Graph()
##  if dummy_input is None:
##    dummy_input = np.random.random_sample((batch_size,224,224,3))
  imageCounter = 0
  outlist=[]
  with g.as_default():
    imagenstack = tf.constant([""])
    imageString=[]
    for imageName in sorted(glob.glob(image_folder + '/*.JPEG')):
      imageString.append(imageName)
      imageCounter = imageCounter + 1
    imagenstack = tf.stack(imageString)

    dataset = tf.data.Dataset.from_tensor_slices(imagenstack)  
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    dataset=dataset.repeat()
    iterator=dataset.make_one_shot_iterator()
    next_element=iterator.get_next()

    out = tf.import_graph_def(
      graph_def=gdef,
      input_map={"input":next_element},
      return_elements=[ "resnet_v2_101/predictions/Softmax"]
    )
    out = out[0].outputs[0]
    print("\n\n image out",out,"\n\n")
    outlist.append(out)
    print("\n\n image out",outlist[-1],"\n\n")
    
  timings=[]
  
  with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      num_iters= int(math.ceil(imageCounter/batch_size))
      print("\n\n\nNumber of Iterations = ",num_iters)
      nvidiasmiCommand = "nohup nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -l 1 -f " + nvidiasmi + " &"
      nmonCommand = "nmon -s1 -c 2000 -F " + nvidiasmi + ".nmon  &"
      pmonCommand = "nohup nvidia-smi pmon -f " + nvidiasmi + ".pmon &"
      os.system(nvidiasmiCommand)
      #os.system(nmonCommand)
      #os.system(pmonCommand)
      tstart=time.time()
      if os.path.exists(latencyF):
          append_write = 'a' # append if already exists
      else:
          append_write = 'w' # make a new file if not
      runtimeResults = open(latencyF, append_write)
      start_process = time.time()

      for k in range(num_iters):

        tic = time.time()
        val = sess.run(outlist)
        tac = time.time()



        runtimeResults.write(str(tac-tic))
        runtimeResults.write("\n")

        if ((tac-start_process) > StopTime):
            break

        #printing lables
        printLables = 0
        if printLables == 1:
            if os.path.exists('resultLables_ResNetV2_101.txt'):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
            #
            highscore = open('resultLables_ResNetV2_101.txt',append_write)
            for index1 in range (0,len(topX(val[0],f.topN)[1])):
              highscore.write(str(getLabels(labels,topX(val[0],f.topN)[1][index1])))
              highscore.write("\n")
            highscore.close()
        #end for prinlables
      timings.append(time.time()-tstart)
      runtimeResults.close()
      # if os.path.exists('runtimes_ResNetV2_101.txt'):
      #     append_write = 'a' # append if already exists
      # else:
      #     append_write = 'w' # make a new file if not
      #
      # runtimeResults = open('runtimes_ResNetV2_101.txt',append_write)
      # runtimeResults.write(str(batch_size) + ',' + str(timings[-1]))
      # runtimeResults.write("\n")
      # runtimeResults.close()
      os.system("pkill nvidia-smi")
      #os.system("pkill nmon")
      sess.close()
      tf.logging.info("Timing loop done!")
      return timings,True,val[0],None



def topX(arr,X):
  ind=np.argsort(arr)[:,-X:][:,::-1]
  return arr[np.arange(np.shape(arr)[0])[:,np.newaxis],ind],ind


def getLabels(labels,ids):
  return [labels[str(x)] for x in ids]

if "__main__" in __name__:
  P=argparse.ArgumentParser(prog="test")

  P.add_argument('--native',action='store_true')
  P.add_argument('--topN',type=int,default=10)
  P.add_argument('--batch_size',type=int,default=128)
  P.add_argument('--image_folder',type=str,default='images')
  P.add_argument('--nvidiasmi_outputFile',type=str,default='nvidiasmi.out')
  P.add_argument('--latencyFile', type=str, default="latencyFile.txt")
  P.add_argument('--stop_time', type=int, default=100)
  
  f,unparsed=P.parse_known_args()
  print(f.image_folder)
  print("Starting at",datetime.datetime.now())

  with open("labels_2015.json","r") as lf:
    labels=json.load(lf)


  if f.native:
    startTime = time.time()
    timings,comp,valnative,mdstats=timeGraph(getResnet50(),f.batch_size,f.image_folder,f.nvidiasmi_outputFile,f.latencyFile,f.stop_time)
    endTime2 = time.time()

    if os.path.exists('runtimes_ResNetV2_101.txt'):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    runtimeResults = open('runtimes_ResNetV2_101.txt', append_write)
    runtimeResults.write(str(f.batch_size) + ', ' + str(endTime2 - startTime))
    runtimeResults.write("\n")
    runtimeResults.close()


  print("Done timing",datetime.datetime.now())

  sys.exit(0)
