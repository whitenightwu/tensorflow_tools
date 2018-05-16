#-*- coding=utf-8 -*-
#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : import_pb_model.py
## Authors    : ydwu@taurus
## Create Time: 2018-04-25:12:08:13
## Description:
## 
##

import import_pb_to_tensorboard as impb
import sys


INPUT_GRAPH='/home/ydwu/framework/tensorflow/ydwu-quan-2/shwu-mobilenet/result_models/frozen_eval/frozen_eval_graph.pb'
OUTPUT_EVENTS='/tmp/tf-ydwu'

if len(sys.argv) == 3:
    INPUT_GRAPH = sys.argv[1]
    OUTPUT_EVENTS = sys.argv[2]

print "INPUT_GRAPH   = %r" % (INPUT_GRAPH) 
print "OUTPUT_EVENTS = %r" % (OUTPUT_EVENTS)


impb.import_to_tensorboard(INPUT_GRAPH, OUTPUT_EVENTS)
