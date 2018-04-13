#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : tfboard.py
## Authors    : ydwu@aries
## Create Time: 2018-03-08:16:43:08
## Description:
## 
##


# /usr/bin/python2.7 ydwu-pb2tfboard.py 

import tensorflow as tf
from tensorflow.python.platform import gfile

####################### 
# INPUT_GRAPH="/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/tmp/mobilenet_v1_1.0_224/unfrozen_graph.pb"

# INPUT_GRAPH='/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan-2/shwu-mobilenet/result_models/05-frozen_eval/frozen_eval_graph.pb'

# INPUT_GRAPH='/home/ydwu/project/ydwu-quan-2/shwu-mobilenet/result_models/05-frozen_eval/frozen_eval_graph.pb'

INPUT_GRAPH="/home/ydwu/quant_tmp/DW-conv/result_model/quantized_graph.pb"

####################### 

# OUTPUT_EVENTS="/tmp/tf-ydwu"
# OUTPUT_EVENTS="/home/ydwu/project/fake-to-quant/network/fn-graph_transforms"

OUTPUT_EVENTS="/home/ydwu/quant_tmp/DW-conv/result_model"

#######################

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile(INPUT_GRAPH, "rb").read())
_ = tf.import_graph_def(graphdef, name="")
summary_write = tf.summary.FileWriter(OUTPUT_EVENTS, graph)

print("final!!!")

# ## tensorboard --logdir=/tmp/tf-ydwu/
# ## firefox

# ## if .pbtxt, you can directly open by tensorboard!

