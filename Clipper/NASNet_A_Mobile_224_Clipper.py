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
import subprocess
import threading
from threading import Thread
os.environ["CUDA_VISIBLE_DEVICES"]="0" #selects a specific device

stepIncrease = 4
N = 1000
powerReading = [-1] * N
completePowerReading = []


class CountdownTask:

    def __init__(self):
        self._running = True


    def terminate(self):
        self._running = False


    def run(self):
        while self._running:
            powerReading.append(float(os.popen('nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits').read()))
            completePowerReading.append(float(os.popen('nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits').read()))
            time.sleep(0.1)



def _parse_function(filename):
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
  return resized


def getResnet50():
  with gfile.FastGFile("frozen_nasnet_mobile.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


def timeGraph(gdef,batch_size=128,image_folder='images',latencyMS = 30, powerCapW = 50, result_file = "ResultsLog.txt"):
  tf.logging.info("Starting execution")
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
  tf.reset_default_graph()
  g = tf.Graph()
##  if dummy_input is None:
##    dummy_input = np.random.random_sample((batch_size,224,224,3))
  imageCounter = 0
  outlist=[]
  with g.as_default():
    imageString = tf.placeholder(tf.string,name='imageString')
    imagenstack = tf.stack(imageString)
    batch_size_dynamic = tf.placeholder(tf.int64, shape=(), name='batch_size_dynamic')
    dataset = tf.data.Dataset.from_tensor_slices(imagenstack)  
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size_dynamic)
    dataset=dataset.repeat()
    iterator = dataset.make_initializable_iterator()
    next_element=iterator.get_next()
    out = tf.import_graph_def(
      graph_def=gdef,
      input_map={"input":next_element},
      return_elements=[ "final_layer/predictions"]
    )
    out = out[0].outputs[0]
    outlist.append(out)
    
  timings=[]
  
  with tf.Session(graph=g,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()

      skipSize = 0
      processed_image = 0
      StringofImage = []
      for imageName in glob.glob(image_folder + '/*.*'):
          StringofImage.append(imageName)
          imageCounter = imageCounter + 1
      #nvidiasmiCommand = "nohup nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -l 1 -f nvidia.txt &"

      #os.system(nvidiasmiCommand)
      listBatchSizeSteps = [x for x in range(1,256)]
      MAXINDEX = len(listBatchSizeSteps)
      MININDEX = 0
      initialBS = 0
      BestBS = 0
      batchCounter = 0

      ThroughputPerSecond = [-1] * N



      newbatchSize = listBatchSizeSteps[initialBS]
      two_second_start=time.time()
      resultsFile = open(result_file, 'w')
      resultsFile.write("time stamp, Througput (image/sec), power, batch size, BestBS, power cap\n")


      while True:

          
          if (time.time() - two_second_start) > 3 :
              two_second_start = time.time()


              #if power cap is meeting, do not do anything. we use 0.9 as a margin to avoid lots of fluctuations
              if max(powerReading) <= powerCapW and max(powerReading) >= (0.8 * powerCapW):
                  #powerReading[-N:-1] = [0] * (len(powerReading) - 20)

                  powerReading[:] = []
                  pass

              elif max(powerReading) < (0.8 * powerCapW):   # latency less than latency cap
                  powerReading[:] = []
                  if BestBS == MAXINDEX:
                      pass
                  else:
                      BestBS = BestBS + stepIncrease
                      newbatchSize = listBatchSizeSteps[BestBS]


              elif max(powerReading) > powerCapW:          # Power More than Power Cap
                  powerReading[:] = []
                  if BestBS == MININDEX:
                      pass
                      #resultsFile.write("No Possible Solution\n")

                  else:
                      BestBS = int(math.floor(0.9 * BestBS)) # Similar to AIMD of Clipper that reduces with 10%
                      newbatchSize = listBatchSizeSteps[BestBS]

            # For last run to make sure that the batch size is not greater than the remaining number of images
          if (processed_image + newbatchSize) > imageCounter:
              print("Entered If for processed_image")
              newbatchSize = imageCounter - processed_image

          StringImage_2 = []
          #print(len(StringofImage))
          for countertemp in range(newbatchSize):
              StringImage_2.append(StringofImage.pop(0))
          tstart = time.time()

          sess.run(iterator.initializer,
                   feed_dict={batch_size_dynamic: newbatchSize, imageString: StringImage_2})


          val = sess.run(outlist,
                         feed_dict={batch_size_dynamic: newbatchSize, imageString: StringImage_2})

          timings.append(time.time() - tstart)

          # Reading power

          printLables = 0  # SET TO ONE FOR LABELS TO BE PRINTED
          if printLables == 1:
              if os.path.exists('resultLables.txt'):
                  append_write = 'a'  # append if already exists
              else:
                  append_write = 'w'  # make a new file if not
              #
              highscore = open('resultLables.txt', append_write)
              for index1 in range(0, len(topX(val[0], f.topN)[1])):
                  highscore.write(str(getLabels(labels, topX(val[0], f.topN)[1][index1])))
                  highscore.write("\n")
              highscore.close()

          ThroughputPerSecond.append(1000/((timings[-1] * 1000)/newbatchSize)) #first convert the time to milisecond (*1000), then divide by the number of processed image (newbatchsize)

          if len(powerReading) == 0:
              time.sleep(1)
              #maxPower = max(powerReading)
          resultsFile.write(
              str(time.time()) + "," + str(ThroughputPerSecond[-1]) + "," + str(max(powerReading)) + "," + str(newbatchSize)
              +  "," + str(BestBS) + "," + str(powerCapW) + "," +"\n")


          #end of our new code
          processed_image = processed_image + newbatchSize
          #print("processed image = ", processed_image)
          #print("skip size is", skipSize)
          #print("batch size ", newbatchSize, " =  ", timings[-1], " s\n\n")
          if processed_image == imageCounter:
              break
          skipSize = skipSize + newbatchSize

          #fileBatch.close()
      sess.close()
      tf.logging.info("Timing loop done!")
      #os.system("pkill nvidia-smi")
      return timings,True,None



def topX(arr,X):
  ind=np.argsort(arr)[:,-X:][:,::-1]
  return arr[np.arange(np.shape(arr)[0])[:,np.newaxis],ind],ind


def getLabels(labels,ids):
  return [labels[str(x)] for x in ids]

if "__main__" in __name__:
  P=argparse.ArgumentParser(prog="test")
  P.add_argument('--FP32',action='store_true')
  P.add_argument('--FP16',action='store_true')
  P.add_argument('--INT8',action='store_true')
  P.add_argument('--native',action='store_true')
  P.add_argument('--topN',type=int,default=10)
  P.add_argument('--batch_size',type=int,default=128)
  P.add_argument('--power_cap', type=int, default=150)
  P.add_argument('--latency', type=int, default=50)
  P.add_argument('--image_folder',type=str,default='images')
  P.add_argument('--result_file',type=str,default='nvidiasmi.out')
  
  f,unparsed=P.parse_known_args()
  print(f.image_folder)
  print("Starting at",datetime.datetime.now())

  with open("labels_2015.json","r") as lf:
    labels=json.load(lf)


  if f.native:
    startTime = time.time()
    c = CountdownTask()
    t = Thread(target=c.run)
    t.start()
    os.system(
        'nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm --format=csv ,nounits, -l 1 -f  ' + str(f.result_file) + '_nvidiasmi_output.csv &')

    timings,comp,mdstats=timeGraph(getResnet50(),f.batch_size,f.image_folder,f.latency, f.power_cap,f.result_file)

    endTime = time.time()

    os.system('pkill nvidia-smi')

    c.terminate()
    t.join()



    if os.path.exists('runtimes_NASNet_A_Mobile_224_Clipper.txt'):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    runtimeResults = open('runtimes_NASNet_A_Mobile_224_Clipper.txt', append_write)
    runtimeResults.write('Total Execution Time, ' + str(f.result_file) + ', ' +str(endTime - startTime))
    runtimeResults.write("\n")
    runtimeResults.close()


  print("Done timing",datetime.datetime.now())


  sys.exit(0)
