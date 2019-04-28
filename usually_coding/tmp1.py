#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : tmp1.py
## Authors    : ydwu@taurus
## Create Time: 2018-04-13:17:10:16
## Description:
## 
##
import tensorflow as tf
import numpy as np

c=tf.constant(value=1)
#print(assert c.graph is tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())
