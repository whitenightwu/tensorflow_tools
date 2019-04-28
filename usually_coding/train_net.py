# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:58:54 2018
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.slim import nets
from resmodel import model
import resmodel
slim = tf.contrib.slim
learning_rate_orig = 0.03

num_train_image=500
MINI_BATCH_SIZE=10
height = 224
width = 224
num_depth = 3
out_path = r"/tfrecords/train"
n_videos_per_record = 5
n_frames = 32
batchsize = 4

def read_and_decode(filename_queue, n_frames, batchsize):
    """Creates one image sequence"""

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq = []


    # for image_count in range(n_frames):
    feature = {
      'video/encoded': tf.FixedLenFeature([], tf.string),
      'video/format': tf.FixedLenFeature([], tf.string),
      'video/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1],dtype=tf.int64)),
      'video/height': tf.FixedLenFeature([], tf.int64),
      'video/width': tf.FixedLenFeature([], tf.int64),
      'video/depth': tf.FixedLenFeature([], tf.int64),
      }

    features = tf.parse_single_example(serialized_example,
                                       features=feature)

    image_buffer = tf.reshape(features['video/encoded'], shape=[])
        # width = tf.cast(features['video/width'], tf.int64)
    image = tf.decode_raw(image_buffer, tf.uint8)
    image_seq = tf.reshape(image, tf.stack([32, height, width, num_depth]))
        # image = tf.reshape(image, [1, height, width, num_depth])
        # image_seq.append(image)

    # image_seq = tf.concat(image_seq, 0)
    label = tf.cast(features['video/class/label'], tf.int64)
    capacity = 100
    min_after_dequeue = 50
    exampleBatch, labelBatch = tf.train.shuffle_batch([image_seq, label], batch_size=batchsize, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return exampleBatch, labelBatch

def get_next_batch(image_seq_tensor_val, lab, sess):
    """Get a batch set of training data.

    Args:
        batch_size: An integer representing the batch size.
        ...: Additional arguments.

    Returns:
        images: A 4-D numpy array with shape [batch_size, height, width,
            num_channels] representing a batch of images.
        labels: A 1-D numpy array with shape [batch_size] representing
            the groundtruth labels of the corresponding images.
    """
    video = sess.run([image_seq_tensor_val])
    label = sess.run(lab)
    v = np.asarray(video)
    v.resize([batchsize, 32, 224, 224, 3])
    v = v/255.0
    l = np.asarray(label)
    l.resize([batchsize])
    return v, l


if __name__ == '__main__':
    filenames = tf.gfile.Glob(os.path.join(out_path, '*.tfrecords'))
    filename_queue_val = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=True)
    image_seq_tensor_val, lab = read_and_decode(filename_queue_val, n_frames, batchsize)
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # maximun alloc gpu50% of MEMconfig.gpu_options.allow_growth = True #allocate dynamically
#    config.gpu_options.allow_growth = True
    num_classes = 5
    num_steps = 10000
    resnet_model_path = r'\train\resnet_v1_50.ckpt'  # Path to the pretrained model
    model_save_path = r'\train\model'  # Path to the model.ckpt-(num_steps) will be saved
    # ...  # Any other constant variables

    inputs = tf.placeholder(tf.float32, shape=[None, 32, 224, 224, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    is_training = tf.placeholder(tf.bool, name='is_training')

    out = model(inputs)
    num_minibatches = int(num_train_image / MINI_BATCH_SIZE)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=labels)
    cost = tf.reduce_mean(loss)
    with tf.name_scope("train"):
#        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_orig).minimize(cost)

        global_steps = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, batchsize * 10, 0.8, staircase=True) 
        train= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#        train = tf.train.AdadeltaOptimizer(learning_rate=learning_rate_orig).minimize(cost)

       # global_steps = tf.Variable(0)
       # learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, batchsize * 10, 0.1, staircase=True)
       # train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

    # with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
    #     net, endpoints = nets.resnet_v1.resnet_v1_50(inputs, num_classes=None,
    #                                                  is_training=is_training)
    #
    # with tf.variable_scope('Logits'):
    #     net = tf.squeeze(net, axis=[1, 2])
    #     net = slim.dropout(net, keep_prob=0.5, scope='scope')
    #     logits = slim.fully_connected(net, num_outputs=num_classes,
    #                                   activation_fn=None, scope='fc')
    # loss = tf.reduce_mean(losses)

        logits = tf.nn.softmax(out)
        classes = tf.argmax(logits, axis=1, name='classes')
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(classes, labels), dtype=tf.float32))


    # init = tf.global_variables_initializer()

    # saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver()
    loss_list = []
    acc_list = []
    loss = 0
    acc = 0
    j = 0
    # config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config = config) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Load the pretrained checkpoint file xxx.ckpt
        # saver.restore(sess, r'\train\resnet_v1_50.ckpt')
        k = 1
        for i in range(40000):
            images, groundtruth_lists = get_next_batch(image_seq_tensor_val, lab, sess)
           # print(np.shape(images))
           # print(groundtruth_lists)
            train_dict = {inputs: images,
                          labels: groundtruth_lists,
                          is_training: True}

            sess.run(train, feed_dict=train_dict)

            loss_, acc_ = sess.run([cost, accuracy], feed_dict=train_dict)
            if j<3000 :
                loss += loss_
                acc +=acc_
                j += 1
            else:
                print("第{}轮loss:{},第{}轮准确率:{}".format(k,loss/3000, k, acc/3000))
                f = open('write.txt','a')
                f.write("第{}轮loss:{},第{}轮准确率:{}\n".format(k,loss/3000, k,acc/3000))
                f.close()
                loss_list.append(loss/3000)
                acc_list.append(acc/3000)
                loss = 0
                acc = 0
                j = 0
                k += 1
                
#            if i % 100 == 0:
            train_text = 'Step: {}, Loss: {}, Accuracy: {}'.format(
                    i + 1, loss_, acc_)
            print(train_text)
            if i % 1000 == 0:
                saver.save(sess, '/data/zhengrui/model/mymodel_{}'.format(i+1))
            # if (i + 1) % 1000 == 0:
            #     saver.save(sess, model_save_path, global_step=i + 1)
            #     print('save mode to {}'.format(model_save_path))
        coord.request_stop()
        coord.join(threads)
        resmodel.save_npy(sess, './model/temp-model.npy')
        print(loss_list,acc_list)
