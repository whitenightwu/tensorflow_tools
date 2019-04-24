#!/usr/bin/env python3
#-*- coding=utf-8 -*-

import os
import numpy as np
import glob
import sys

import tensorflow as tf

slim = tf.contrib.slim
from tensorflow.python.framework import importer
from tensorflow.python.framework import graph_util


if len(sys.argv) == 3:
    INPUT_MODEL = sys.argv[1]
    num_examples = sys.argv[2] 
else:    
    INPUT_MODEL = '/home/ydwu/project/mobilenet-quant-Q_DW/graph_transforms-Q_DW/result_model/quantized_graph.pb'
    num_examples=100

    
dataset_tfRecord = '/mllib/ImageNet/tfrecords/validation-00077-of-00128'

#####################################################################

def read_tfRecord(file_tfRecord):

    queue = tf.train.string_input_producer([file_tfRecord])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(queue)
    image_features = tf.parse_single_example( serialized_example, features={'image/encoded':tf.VarLenFeature(tf.string),  'image/height': tf.FixedLenFeature([], tf.int64), 'image/width':tf.FixedLenFeature([], tf.int64),  'image/class/label': tf.FixedLenFeature([], tf.int64), 'image/channels': tf.FixedLenFeature([], tf.int64)})

    height = tf.cast(image_features['image/height'], tf.int32)
    width = tf.cast(image_features['image/width'], tf.int32)
    channels = tf.cast(image_features['image/channels'], tf.int32)

    label = tf.cast(image_features['image/class/label'], tf.int64)
    
    image_shape = tf.parallel_stack([height, width, channels])
    image = tf.sparse_tensor_to_dense(image_features['image/encoded'], default_value='<PAD>')
    image = tf.image.decode_jpeg(image[0], channels=3)


    # # ydwu - mobilenet
    # image = tf.reshape(image, image_shape)
    # image = tf.image.resize_images(image, (224,224))
    # image = tf.reshape(image,[224,224,3])
    # image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)

    # mobilenet
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
    image = tf.squeeze(image, [0])
      
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    return image,label,height,width


#####################################################################


if __name__ == '__main__':

    ### load .pb
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(INPUT_MODEL,'rb') as f:        
        graph_def.ParseFromString(f.read())
        _ = importer.import_graph_def(graph_def, name="")
        image,label,height,width = read_tfRecord(dataset_tfRecord)

        
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        input_x = sess.graph.get_tensor_by_name("input:0")
        out_acc = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')        
        out_sss = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Softmax:0')
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
            
        count=0                
        for i in range(num_examples):            
            img,lab,width_print,height_print = sess.run([image, label,width,height])            
            # print(height_print)

            img = tf.reshape(img,[1,224,224,3])
            img = img.eval()
            
            img_out_softmax = sess.run(out_acc, feed_dict={input_x:img})            
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            # print ("prediction label:",prediction_labels)
            # print('true label:',lab)
            correct_prediction_2 = tf.equal(lab, prediction_labels)             
            print("correct_prediction:", correct_prediction_2.eval()[0])
            if correct_prediction_2.eval()[0] == 1:
                count+=1

        print('count:', count)
        print('accurcy:%.2f'%(float(count)/float(num_examples)))
        coord.request_stop()
        coord.join(threads)
