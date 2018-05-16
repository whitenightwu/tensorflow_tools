#!/usr/bin/env python
#-*- coding=utf-8 -*-

import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import model

ckpt_filename = './model/PlatesRec_mobilenet_20171108-928-finetune.ckpt-6190000'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
isess = tf.InteractiveSession()

# Input placeholder.

img_input = tf.placeholder(tf.float32, shape=(40, 120, 3))
input_data = tf.expand_dims(img_input, 0)

y, _ = model.mobilenet(input_data, width_multiplier=1.0,stride_in = 1, size = 3, is_training=False)
best = tf.argmax(tf.reshape(y, [-1, 8, len(model.chars_klyu)]), 2)  #best shape (?, 7)

isess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
#print ckpt_filename
saver.restore(isess, ckpt_filename)

# Test on some demo image and visualize output.
from tensorflow.python.framework import graph_io
#img = mpimg.imread(path + 'test_rose_1.jpg')
#logits_rn =  process_image(img)
saver.save(isess,'./mobilenet-freeze_and_quantize/ckpt_graph/platerec_model_graph_10928f2_reshape.ckpt')
graph_io.write_graph(isess.graph, './mobilenet-freeze_and_quantize/ckpt_graph','platerec_model_graph_10928f2_reshape.pbtxt')


