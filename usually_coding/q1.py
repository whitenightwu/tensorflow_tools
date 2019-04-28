# -- coding: utf-8 --
#!/usr/local/bin/python3
##        (C) COPYRIGHT Ingenic Limited.
##             ALL RIGHTS RESERVED
##
## File       : q1.py
## Authors    : ydwu@aries
## Create Time: 2018-03-14:19:43:07
## Description: linear regression

## 
##

import os
import sys

import tensorflow as tf
import numpy as np
#定义数据
print("111")
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
#构造方程式的数据y=0.1x+0.2x+0.3
y_data = np.dot([0.100, 0.200], x_data) + 0.300
#定义变量
x_data

b = tf.Variable(tf.zeros([1]))
#w1 w2,b
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
#定义公式
y = tf.matmul(W, x_data) + b
#定义损失函数
loss = tf.reduce_mean(tf.square(y - y_data))
#0.5是学习率,选择参数更新方式是随机梯度
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#初始化变量
init = tf.initialize_all_variables()

#启动图
sess = tf.Session()
sess.run(init)

#拟合
for step in range(0, 20100):
   sess.run(train)
   #每二十步打印一次
   if step % 20 == 0:
       print( "step:",step, "loss:",sess.run(loss),sess.run(W), sess.run(b))
