#!/bin/bash

##################################################

# INPUT_MODEL=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/mobilenet-freeze_and_quantize/frozen_graph/platerec_model_graph_10928f2_reshape.pb

# python ./optimize_for_inference.py \
# --input=${INPUT_MODEL} \
# --output=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/result/merge-BN.pb \
# --frozen_graph=True \
# --input_names=Placeholder \
# --output_names=ArgMax





##################################################

# INPUT_MODEL=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/merge-BN/tmp/mobilenet_v1_1.0_224/unfrozen_graph.pb


INPUT_MODEL=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/tmp/mobilenet_v1_1.0_224/frozen_graph.pb

python ./optimize_for_inference.py \
--input=${INPUT_MODEL} \
--output=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/result/merge-BN.pb \
--frozen_graph=True \
--input_names=input \
--output_names=MobilenetV1/Predictions/Reshape_1



sleep 1
echo ydwu == ok-end
