#!/usr/bin/env python
#-*- coding=utf-8 -*-

import import_pb_to_tensorboard as impb

impb.import_to_tensorboard("/home/ydwu/framework/tensorflow/ydwu-quan-2/shwu-mobilenet/result_models/frozen_eval/frozen_eval_graph.pb", "logs_test")
