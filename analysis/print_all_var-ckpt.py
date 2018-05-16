# -*- coding: utf-8 -*-
#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : qqq.py
## Authors    : ydwu@taurus
## Create Time: 2018-04-02:15:33:23
## Description:
## 
## 查看ckpt保存的变量及名字
import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow

#######################################################
################## ckpt
# logdir='./output/'

# ckpt_dir='/home/ydwu/framework/tensorflow/ydwu-quan-2/shwu-mobilenet/result_models/20180402-164645/model-20180402-164645.ckpt-2293'

# ckpt_dir='/home/ydwu/project/ydwu-quan-2/shwu-mobilenet/result_models/20180409-114811/model-20180409-114811.ckpt-17704'

ckpt_dir='/home/ydwu/framework/tensorflow-models/research/slim/nets/ydwu-mobilenet/quan-train/model.ckpt-129'

print(ckpt_dir)

reader = pywrap_tensorflow.NewCheckpointReader(ckpt_dir)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))



