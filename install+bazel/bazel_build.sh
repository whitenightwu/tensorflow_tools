#!/usr/bin/env bash
error bash

####
pip uninstall tensorflow
pip uninstall tensorboard
bazel clean

####
./configure (/home/ydwu/anaconda2/lib/python2.7/site-packages)

####
bazel build --config=opt  //tensorflow/tools/pip_package:build_pip_package
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

####
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/
# pip install /tmp/tensorflow_pkg/tensorflow-1xxxxxxx.whl


####
(1)
bazel build tensorflow/tools/graph_transforms:transform_graph
or
bazel build tensorflow/contrib/quantize/python:quantize_graph
or
bazel build tensorflow/tools/quantization:quantize_graph

(2)
bazel build tensorflow/examples/label_image:label_image

(3)
bazel build tensorflow/tools/graph_transforms:summarize_graph

(4)
bazel build --config=opt tensorflow/core/kernels:conv_ops
bazel build -c opt --jobs 1 //tensorflow/cc:tutorials_example_trainer

####
sleep 1
