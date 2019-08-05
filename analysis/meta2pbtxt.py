#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : meta2pbtxt.py
## Authors    : whitewu@whitewu-ubuntu
## Create Time: 2019-08-05:12:05:58
## Description:
## 
##
import os
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
