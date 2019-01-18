#!/usr/bin/python3

####
# Rill create for tf test at 2017-01-10
####


import tensorflow as tf
import numpy as np

#gen rand data
x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.100,0.200],x_data) + 0.300

#build liner model
b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(w,x_data) + b

#minimize square
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.3)
train = optimizer.minimize(loss)

#init variable
init = tf.initialize_all_variables()

#start graph
sess = tf.Session()
sess.run(init)

#fit
for step in range(0,601):
    sess.run(train)
    if (step % 20) == 0:
        print(step , sess.run(w) , sess.run(b))









