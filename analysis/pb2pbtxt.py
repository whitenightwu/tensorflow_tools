#!/usr/bin/env python
#coding:utf-8
# MIT License
# 
# Copyright (c) 2016



import sys  
from datetime import datetime
import os.path
import time
import numpy as np
import importlib
import itertools
import argparse
from six.moves import xrange

sys.path.append("/home/ydwu/work/facenet/src")
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops

###############################
# network = importlib.import_module("src.models.squeezenet")
# print(network)

# # /home/ydwu/work/facenet/src/models/squeezenet.py
# image = tf.placeholder(shape=[160,160,3], dtype=tf.float32)
# image = tf.image.per_image_standardization(image)
# image = tf.reshape(image, [1, 160,160,3])
#                 # file_contents = tf.read_file(filename)
#                 # image = tf.image.decode_image(file_contents, channels=3)
                
#                 # if args.random_crop:
#                 #     image = tf.random_crop(image, [args.image_size, args.image_size, 3])
#                 # else:
#                 #     image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
#                 # if args.random_flip:
#                 #     image = tf.image.random_flip_left_right(image)
    
#                 # #pylint: disable=no-member
#                 # image.set_shape((args.image_size, args.image_size, 3))

# # Build the inference graph
# prelogits, _ = network.inference(image, 1.0, 
#                                  phase_train=False, bottleneck_layer_size=128)
        
# embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

# graph = tf.get_default_graph()
# # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
# sess = tf.Session()#config=tf.ConfigProto()#gpu_options=gpu_options))        


# tf.train.write_graph(graph, '/tmp/tf-ydwu/','graph.pbtxt')

###############################
###############################
###############################
import tensorflow as tf
from tensorflow.python.platform import gfile
import sys


INPUT_GRAPH='/home/ydwu/project/Optimize_Model_Tools/mobilenet/model-20180928-093249.ckpt-186752.meta'
OUTPUT_EVENTS='/tmp/tf-ydwu'

if len(sys.argv) == 3:
    INPUT_GRAPH = sys.argv[1]
    OUTPUT_EVENTS = sys.argv[2]

print "INPUT_GRAPH   = %r" % (INPUT_GRAPH) 
print "OUTPUT_EVENTS = %r" % (OUTPUT_EVENTS)

graph = tf.get_default_graph()
_ = tf.train.import_meta_graph(INPUT_GRAPH)
summary_write = tf.summary.FileWriter(OUTPUT_EVENTS , graph)
tf.train.write_graph(graph, '/tmp/tf-ydwu/','graph.pbtxt', as_text=True)

# ## tensorboard --logdir=/tmp/tf-ydwu/
# ## firefox
