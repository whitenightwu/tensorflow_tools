
#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : ydwu-ckpt2tfboard.py
## Authors    : ydwu@taurus
## Create Time: 2018-04-08:15:56:27
## Description:
## 
##

import tensorflow as tf
from tensorflow.python.platform import gfile
import sys


INPUT_GRAPH='/home/ydwu/project/ydwu-quan-2/shwu-mobilenet/result_models/04-eval/tmp.ckpt.meta'
OUTPUT_EVENTS='/tmp/tf-ydwu'

if len(sys.argv) == 3:
    INPUT_GRAPH = sys.argv[1]
    OUTPUT_EVENTS = sys.argv[2]

print "INPUT_GRAPH   = %r" % (INPUT_GRAPH) 
print "OUTPUT_EVENTS = %r" % (OUTPUT_EVENTS)

graph = tf.get_default_graph()
_ = tf.train.import_meta_graph(INPUT_GRAPH)
summary_write = tf.summary.FileWriter(OUTPUT_EVENTS , graph)


# ## tensorboard --logdir=/tmp/tf-ydwu/
# ## firefox
