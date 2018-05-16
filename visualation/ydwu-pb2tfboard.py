#!/usr/bin/python2.7
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : tfboard.py
## Authors    : ydwu@aries
## Create Time: 2018-03-08:16:43:08
## Description:
##    transform "*.pb" into "events.out.tfevents.*"
##


# /usr/bin/python2.7 ydwu-pb2tfboard.py 

import tensorflow as tf
from tensorflow.python.platform import gfile
import sys

if len(sys.argv) == 3:
    INPUT_GRAPH = sys.argv[1]
    OUTPUT_EVENTS = sys.argv[2]

else:    
    ####################### 
    # INPUT_GRAPH="/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/tmp/mobilenet_v1_1.0_224/unfrozen_graph.pb"
    # INPUT_GRAPH='/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan-2/shwu-mobilenet/result_models/05-frozen_eval/frozen_eval_graph.pb'

    # INPUT_GRAPH="/home/ydwu/project/mobilenet-quant-Q_DW/graph_transforms-Q_DW/result_model/quantized_graph.pb"
    # INPUT_GRAPH="/home/ydwu/project/mobilenet-quant-Q_DW/quantization-Q_DW/result_model/quantization.pb"

    INPUT_GRAPH="/home/ydwu/project/freeze_graph/frozen_eval_graph.pb"

    ####################### 
    
    # OUTPUT_EVENTS="/tmp/tf-ydwu"
    
    # OUTPUT_EVENTS="/home/ydwu/project/fake-to-quant/network/fn-graph_transforms"
    # OUTPUT_EVENTS="/home/ydwu/project/facenet-quant/network/fn-graph_transforms"
    
    # OUTPUT_EVENTS="/home/ydwu/project/mobilenet-quant-Q_DW/graph_transforms-Q_DW/result_model"
    # OUTPUT_EVENTS="/home/ydwu/project/mobilenet-quant-Q_DW/quantization-Q_DW/result_model"
    
    OUTPUT_EVENTS="/home/ydwu/project/freeze_graph"
    
####################### 

print "INPUT_GRAPH   = %r" % (INPUT_GRAPH) 
print "OUTPUT_EVENTS = %r" % (OUTPUT_EVENTS)
# print("INPUT_GRAPH = ", INPUT_GRAPH)
# print("OUTPUT_EVENTS = ", OUTPUT_EVENTS)

#######################

graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile(INPUT_GRAPH, "rb").read())
_ = tf.import_graph_def(graphdef, name="")
summary_write = tf.summary.FileWriter(OUTPUT_EVENTS, graph)

print("complete!!!")

summary_write.close()


# ## tensorboard --logdir=/tmp/tf-ydwu/
# ## firefox

# ## if .pbtxt, you can directly open by tensorboard!

