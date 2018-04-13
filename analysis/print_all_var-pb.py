# -*- coding: utf-8 -*-
#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : print_all_var-pb.py
## Authors    : ydwu@taurus
## Create Time: 2018-04-10:14:14:53
## Description:
## 
## 查看pb保存的变量及名字
import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow


## using dirctly summary_graph.sh!!!!!!!!

#######################################################
################## pb


output_graph_path='/home/ydwu/project/fake-to-quant/network/quantized_graph.pb'





with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:


        ########  plan A
        # sess.run(tf.global_variables_initializer())
        # for variable_name in tf.all_variables():
        # # for variable_name in tf.global_variables():
        #     # variable_name = [v.name for c in tf.all_variables()]
        #     print("======================")
        #     shape = variable_name.get_shape()
        #     print(variable_names)

        # ########  plan B
        # ops = sess.graph.get_operations()
        # for op in ops:
        #     # print(op.name)
        #     # print(op.values)
        #     ydwu_tensor = op.outputs[0]
        #     # print(ydwu_tensor)
        #     print('tensor.name = ', ydwu_tensor.name)
        #     print(ydwu_tensor.eval())


        ########  plan C
        import re
        ops = sess.graph.get_operations()
        for op in ops:
            if op.type == 'Const':
                # pattern=re.compile(r'^max')
                # if re.match(pattern, op.name):
                if re.search(r'(max)|(min)', op.name) :
                    print(op.name)
                    ydwu_tensor = op.outputs[0]
                    print(tf.contrib.util.constant_value(ydwu_tensor))

                
# ydwu_tensor = op.outputs[0].name
# print(sess.graph.get_tensor_by_name(ydwu_tensor))

        
#### C++ api
# num_elements = ydwu_tensor.NumElements()
# ydwu_values = ydwu_tensor.flat<float>().data()            
