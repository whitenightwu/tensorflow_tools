import tensorflow as tf
import math
import numpy as np
is_training = True
var_dict = {}
data_dict = None
trainable = True
label_num = 101

def save_npy(sess, npy_path="./model/Resnet-save.npy"):
    data_dict = {}

    for (name, idx), var in list(var_dict.items()):
        var_out = sess.run(var)
        if name not in data_dict:
           data_dict[name] = {}
        data_dict[name][idx] = var_out
    np.save(npy_path, data_dict)
    print(("file saved", npy_path))
    return npy_path

def get_fc_var(in_size, out_size, name):
    """
    in_size : number of input feature size
    out_size : number of output feature size
    name : block_layer name
    """
    initial_value = tf.truncated_normal([in_size, out_size], 0.0, stddev=1 / math.sqrt(float(in_size)))
    weights = get_var(initial_value, name, 0, name + "_weights")

    initial_value = tf.truncated_normal([out_size], 0.0, 1.0)
    biases = get_var(initial_value, name, 1, name + "_biases")

    return weights, biases


def get_var(initial_value, name, idx, var_name):
    """
    load variables from Loaded model or new generated random variables
    initial_value : random initialized value
    name: block_layer name
    index: 0,1 weight or bias
    var_name: name + "_filter"/"_bias"
    """
    if ((name, idx) in var_dict):
        print("Reuse Parameters...")
        print(var_dict[(name, idx)])
        return var_dict[(name, idx)]

    if data_dict is not None and name in data_dict:
        value = data_dict[name][idx]
    else:
        value = initial_value

    if trainable:
        var = tf.Variable(value, name=var_name)
    else:
        var = tf.constant(value, dtype=tf.float32, name=var_name)

    var_dict[(name, idx)] = var

    # print var_name, var.get_shape().as_list()
    assert var.get_shape() == initial_value.get_shape()

    return var


def res_block_3_layers(bottom, channel_list, name, change_dimension=False, block_stride=1):
    """
    bottom: input values (X)
    channel_list : number of channel in 3 layers
    name: block name
    """
    if (change_dimension):
        short_cut_conv = conv_layer(bottom, 1, 1, bottom.get_shape().as_list()[-1], channel_list[2], 1, block_stride,
                                         name + "_ShortcutConv")
        block_conv_input = batch_norm(short_cut_conv)
    else:
        block_conv_input = bottom

    block_conv_1 = conv_layer(bottom, 3, 1, bottom.get_shape().as_list()[-1], channel_list[0], 1, block_stride,
                                   name + "_lovalConv1")
    block_norm_1 = batch_norm(block_conv_1)
    block_elu_1 = tf.nn.elu(block_norm_1)

    block_conv_2 = conv_layer(block_elu_1, 1, 3, channel_list[0], channel_list[1], 1, 1, name + "_lovalConv2")
    block_norm_2 = batch_norm(block_conv_2)
    block_elu_2 = tf.nn.elu(block_norm_2)

    block_conv_3 = conv_layer(block_elu_2, 1, 1, channel_list[1], channel_list[2], 1, 1, name + "_lovalConv3")
    block_norm_3 = batch_norm(block_conv_3)
    block_res = tf.add(block_conv_input, block_norm_3)
    elu = tf.nn.elu(block_res)

    return elu


def max_pool(bottom, kernal_size1=1, kernal_size2=2, stride1=1, stride2=2, name="max"):
    """
    bottom: input values (X)
    kernal_size : n * n kernal
    stride : stride
    name : block_layer name
    """
    print(name + ":")
    print(bottom.get_shape().as_list())
    return tf.nn.max_pool3d(bottom, ksize=[1,kernal_size1, kernal_size2, kernal_size2, 1], strides=[1, stride1, stride2, stride2, 1],
                          padding='SAME', name=name)

def avg_pool(bottom,kernal_size1=1, kernal_size2 = 2,stride1=1, stride2 = 2, name = "avg"):
    """
    bottom: input values (X)
    kernal_size : n * n kernal
    stride : stride
    name : block_layer name
    """
    print(name + ":")
    print(bottom.get_shape().as_list())
    return tf.nn.avg_pool3d(bottom, ksize=[1,kernal_size1, kernal_size2, kernal_size2, 1], strides=[1,stride1, stride2, stride2, 1], padding='VALID', name=name)

def batch_norm(inputsTensor):
    """
    Batchnorm
    """
    _BATCH_NORM_DECAY = 0.99
    _BATCH_NORM_EPSILON = 1e-12
    return tf.layers.batch_normalization(inputs=inputsTensor, axis=-1, momentum=_BATCH_NORM_DECAY,
                                         epsilon=_BATCH_NORM_EPSILON, center=True, scale=True,
                                         training=is_training)


def get_conv_var(filter_size1, filter_size2, in_channels, out_channels, name):
    """
    filter_size : 3 * 3
    in_channels : number of input filters
    out_channels : number of output filters
    name : block_layer name
    """
    initial_value = tf.truncated_normal([filter_size1, filter_size2, filter_size2, in_channels, out_channels], 0.0,
                                        stddev=1 / math.sqrt(float(filter_size2 * filter_size2)))
    filters = get_var(initial_value, name, 0, name + "_filters")

    initial_value = tf.truncated_normal([out_channels], 0.0, 1.0)
    biases = get_var(initial_value, name, 1, name + "_biases")

    return filters, biases


def conv_layer(bottom, kernal_size1, kernal_size2, in_channels, out_channels, stride1, stride2, name):
    """
    bottom: input values (X)
    kernal_size : n * n kernal
    in_channels: number of input filters
    out_channels : number of output filters
    stride : stride
    name : block_layer name
    """
    print(name + ":")
    print(bottom.get_shape().as_list())
    with tf.variable_scope(name):
        filt, conv_biases = get_conv_var(kernal_size1, kernal_size2, in_channels, out_channels, name)

        conv = tf.nn.conv3d(bottom, filt, [1, stride1, stride2, stride2, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, conv_biases)

        tf.summary.histogram('weight', filt)
        tf.summary.histogram('bias', conv_biases)

        return bias


def fc_layer(bottom, in_size, out_size, name):
    """
    bottom: input values (X)
    in_size : number of input feature size
    out_size : number of output feature size
    """
    print(name + ":")
    print(bottom.get_shape().as_list())
    with tf.variable_scope(name):
        weights, biases = get_fc_var(in_size, out_size, name)

        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        tf.summary.histogram('weight', weights)
        tf.summary.histogram('bias', biases)

        return fc


def model(x):
    conv1 = conv_layer(x, 5, 7, 3, 64, 1, 2, "conv1")
    conv_norm_1 = batch_norm(conv1)
    conv1_elu = tf.nn.elu(conv_norm_1)
    pool1 = max_pool(conv1_elu, 1, 3, 1, 2, "pool1")

    block1_1 = res_block_3_layers(pool1, [64, 64, 256], "block1_1", True)
    block1_2 = res_block_3_layers(block1_1, [64, 64, 256], "block1_2")
    block1_3 = res_block_3_layers(block1_2, [64, 64, 256], "block1_3")

    block2_1 = res_block_3_layers(block1_3, [128, 128, 512], "block2_1", True, 2)
    block2_2 = res_block_3_layers(block2_1, [128, 128, 512], "block2_2")
    block2_3 = res_block_3_layers(block2_2, [128, 128, 512], "block2_3")
    block2_4 = res_block_3_layers(block2_3, [128, 128, 512], "block2_4")

    block3_1 = res_block_3_layers(block2_4, [256, 256, 1024], "block3_1", True, 2)
    block3_2 = res_block_3_layers(block3_1, [256, 256, 1024], "block3_2")
    block3_3 = res_block_3_layers(block3_2, [256, 256, 1024], "block3_3")
    block3_4 = res_block_3_layers(block3_3, [256, 256, 1024], "block3_4")
    block3_5 = res_block_3_layers(block3_4, [256, 256, 1024], "block3_5")
    block3_6 = res_block_3_layers(block3_5, [256, 256, 1024], "block3_6")

    block4_1 = res_block_3_layers(block3_6, [512, 512, 2048], "block4_1", True, 2)
    block4_2 = res_block_3_layers(block4_1, [512, 512, 2048], "block4_2")
    block4_3 = res_block_3_layers(block4_2, [512, 512, 2048], "block4_3")

    pool2 = avg_pool(block4_3, 32, 7, 1, 1, "pool2")

    fc1 = fc_layer(pool2, 2048, label_num, "fc1200")
    fc2 = fc_layer(fc1, 1024, label_num, "fc")
    return fc1

