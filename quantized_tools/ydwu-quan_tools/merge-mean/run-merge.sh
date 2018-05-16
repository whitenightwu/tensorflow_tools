#!/bin/bash



# --output=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/result/merge-BN.pb \

##################################################

# INPUT_MODEL=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/mobilenet-freeze_and_quantize/frozen_graph/platerec_model_graph_10928f2_reshape.pb

# python ./ydwu_merge.py \
# --input=${INPUT_MODEL} \
# --output=./tmp/merge-BN.pb \
# --frozen_graph=True \
# --input_names=Placeholder \
# --output_names=ArgMax \
# --merge_bn=True \
# --merge_bn_mode=ydwu1 \
# --merge_pre=False




##################################################


INPUT_MODEL=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/tmp/mobilenet_v1_1.0_224/frozen_graph.pb

python ./ydwu_merge.py \
--input=${INPUT_MODEL} \
--output=./tmp/merge-BN.pb \
--frozen_graph=True \
--input_names=input \
--output_names=MobilenetV1/Predictions/Reshape_1 \
--merge_bn=True \
--merge_bn_mode=ydwu2 \
--merge_pre=True \
--input_mean=127 \
--input_std=127
# True
# False

sleep 1
echo ydwu == ok-end
