



#!/usr/bin/env python3
#-*- coding=utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import sys  

#### <module 'tensorflow' from '/usr/local/lib/python2.7/dist-packages/tensorflow/__init__.pyc'>

sys.path.append("/home/ydwu/framework/tensorflow22/tensorflow/tensorflow")
# sys.path.append("/home/ydwu/framework/tensorflow22/tensorflow")

# sys.path.append("/home/ydwu/framework/tensorflow22/tensorflow/bazel-tensorflow")
# sys.path.append("/home/ydwu/framework/tensorflow22/tensorflow/bazel-tensorflow/tensorflow")


import tensorflow as tf
print(tf)

slim = tf.contrib.slim
from tensorflow.python.framework import importer
from tensorflow.python.framework import graph_util

dataset_tfRecord = '/mllib/ImageNet/tfrecords/validation-00001-of-00128'
# dataset_tfRecord = '/mllib/ImageNet/tfrecords/validation-00004-of-00128'

batch_size=96

graph_path = '/home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/scripts/mobilenet/tmp/mobilenet_v1_1.0_224/frozen_graph.pb'

# graph_path = '/home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/scripts/mobilenet/tmp/mobilenet_v1_1.0_224/unfrozen_graph.pb'

# graph_path = '/home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/scripts/mobilenet/tmp/mobilenet_v1_1.0_224/quantized_graph.pb'



### load tfrecord
def read_tfRecord(file_tfRecord):
    queue = tf.train.string_input_producer([file_tfRecord])
    print("ydwu=========read_tfRecord===111")
    print(graph_path)
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(queue)
    # print(aa_)
    # print(serialized_example)

    #############################

    # features={'image/encoded':tf.FixedLenFeature([], tf.string)
    image_features = tf.parse_single_example( serialized_example, features={'image/encoded':tf.VarLenFeature(tf.string),  'image/height': tf.FixedLenFeature([], tf.int64), 'image/width':tf.FixedLenFeature([], tf.int64),  'image/class/label': tf.FixedLenFeature([], tf.int64), 'image/channels': tf.FixedLenFeature([], tf.int64)})

    #############################


    # height = tf.cast(image_features['image/height'], tf.int64)
    # width = tf.cast(image_features['image/width'], tf.int64)
    # channels = tf.cast(image_features['image/channels'], tf.int64)

    height = tf.cast(image_features['image/height'], tf.int32)
    width = tf.cast(image_features['image/width'], tf.int32)
    channels = tf.cast(image_features['image/channels'], tf.int32)

    ################# get image value

    print(image_features['image/encoded'])
    
    image_shape = tf.parallel_stack([height, width, channels])

    image = tf.sparse_tensor_to_dense(image_features['image/encoded'], default_value='<PAD>')

    image = tf.image.decode_jpeg(image[0], channels=3)

    # image = tf.decode_raw(image_features['image/encoded'],tf.uint8)
    # image = tf.decode_raw(image, tf.uint8)

    image = tf.reshape(image, image_shape)
    image = tf.image.resize_images(image, (224,224))
    image = tf.reshape(image,[224,224,3])

    # # image = tf.random_crop(image, [224, 224, 3])
    # # distorted_image = tf.image.random_flip_left_right(distorted_image)
    # # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # # distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # # image = tf.image.resize_images(image, (224,224))

    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    # image = (image -127)/127
    ################################################3

    label = tf.cast(image_features['image/class/label'], tf.int64)


    # print(image)
    # print(label)
    # print(features['image/channels'])

    return image,label,height,width






if __name__ == '__main__':

    # print(tf.__path__)

    # images, sparse_labels = tf.train.shuffle_batch(
    #     [image, label],
    #     batch_size=50,
    #     num_threads=2,
    #     capacity=1000 + 3 * 50,
    #     min_after_dequeue=1000)

        ### load .pb
    print("ydwu============= load .pb")
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(graph_path,'rb') as f:
        
        graph_def.ParseFromString(f.read())
        _ = importer.import_graph_def(graph_def, name="")
            
        image,label,height,width = read_tfRecord(dataset_tfRecord)
            
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # sess.run(tf.initialize_all_variables())
        input_x = sess.graph.get_tensor_by_name("input:0")

        out_acc = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')
        # out_acc = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1_eightbit_reshape_MobilenetV1/Predictions/Softmax:0')

        
        out_sss = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Softmax:0')
        # print(input_x)
        # print(out_acc)
        # print(out_sss)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
            
        print("ydwu========= testing")
        count=0
                
        for i in range(batch_size):
            
            ### load tfrecord
            # print("ydwu============= load tfrecord")
            img,lab,width_print,height_print = sess.run([image, label,width,height])
            # print(height_print)
            # print(width_print)
            # print(img)
            # print(lab)
            
            img = tf.reshape(img,[1,224,224,3])
            img = img.eval()

            print("ydwu==========tmp")
            img_out_softmax = sess.run(out_acc, feed_dict={input_x:img})
            # img_out_softmax = sess.run(out_sss, feed_dict={input_x:img})
            # print ("img_out_softmax:",img_out_softmax)
            print("ydwu==========tmp2222222")
            
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print ("prediction label:",prediction_labels)
            print('true label:',lab)
            correct_prediction_2 = tf.equal(lab, prediction_labels)             
            print("correct_prediction:", correct_prediction_2.eval()[0])
            if correct_prediction_2.eval()[0] == 1:
                count+=1

        # accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
        # print("accuracy_2", accuracy_2.eval()[0])
        print('count:', count)
        print('accurcy:%.2f'%(float(count)/float(batch_size)))
        

        coord.request_stop()
        coord.join(threads)
        sess.close()







#     with tf.Session() as sess:

#         tf.initialize_all_variables().run()
#         input_x = sess.graph.get_tensor_by_name("input:0")
#         print input_x
#         output = sess.graph.get_tensor_by_name("output:0")
#         print output

#         y_conv_2 = sess.run(output,{input_x:mnist.test.images})
#         print "y_conv_2", y_conv_2

#         # Test trained model
#         #y__2 = tf.placeholder("float", [None, 10])
#         y__2 = mnist.test.labels;
#         correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y__2, 1))
#         print "correct_prediction_2", correct_prediction_2
#         accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
#         print "accuracy_2", accuracy_2

#         print "check accuracy %g" % accuracy_2.eval()
