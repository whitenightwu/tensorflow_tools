#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : print_all_var-ckpt-2.py
## Authors    : ydwu@taurus
## Create Time: 2018-06-19:12:22:30
## Description:
## 
##
import os
import sys

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

DIR="/home/ydwu/project/mobilenet-fake/results/quan-train/model.ckpt-937"

print_tensors_in_checkpoint_file(DIR, None, True)
