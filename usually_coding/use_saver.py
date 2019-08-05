#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : use_saver.py
## Authors    : ydwu@ydwu-white
## Create Time: 2019-04-28:11:42:22
## Description:
## 
##

迁移学习的实现需要网络在其他数据集上做预训练，完成参数调优工作，然后拿预训练好的参数在新的任务上做fine-tune，但是有时候可能只需要预训练的网络的一部分权重，本文主要提供一个方法如何在tf上加载想要加载的权重。



1) way one
在使用tensorflow加载网络权重的时候，直接使用tf.train.Saver().restore(sess, ‘ckpt’)的话是直接加载了全部权重，我们可能只需要加载网络的前几层权重，或者只要或者不要特定几层的权重，这时可以使用下面的方法：

var = tf.global_variables()
var_to_restore = [val  for val in var if 'conv1' in val.name or 'conv2'in val.name]
saver = tf.train.Saver(var_to_restore )
saver.restore(sess, os.path.join(model_dir, model_name))
var_to_init = [val  for val in var if 'conv1' not in val.name or 'conv2'not in val.name]
tf.initialize_variables(var_to_init)

这样就只从ckpt文件里只读取到了两层卷积的卷积参数，前提是你的前两层网络结构和名字和ckpt文件里定义的一样。将var_to_restore和var_to_init反过来就是加载名字中不包含conv1、2的权重。




2) way two
如果使用tensorflow的slim选择性读取权重的话就更方便了

exclude = ['layer1', 'layer2']
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, os.path.join(model_dir, model_name))

这样就完成了不读取ckpt文件中’layer1’, ‘layer2’权重

