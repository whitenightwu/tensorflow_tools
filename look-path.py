#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : tf-path.py
## Authors    : ydwu@aries
## Create Time: 2018-02-09:14:03:46
## Description:
## 
##
import os
import sys

import tensorflow as tf
print(tf.__path__)


# Return True if TensorFlow was build with CUDA(GPU)support
# print(tf.test.is_built_with_cuda())
if tf.test.is_built_with_cuda():
    print("The installed version of TensorFlow includes GPU support.")
else:
    print("The installed version of TensorFlow does not includes GPU support.")


print(tf.test.gpu_device_name())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
