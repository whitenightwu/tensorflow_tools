#!/bin/bash

python ./freeze_graph_preprocess.py
sleep 1

python ./tools/freeze_graph.py \
--input_graph=./mobilenet-freeze_and_quantize/ckpt_graph/platerec_model_graph_10928f2_reshape.pbtxt \
--input_checkpoint=./mobilenet-freeze_and_quantize/ckpt_graph/platerec_model_graph_10928f2_reshape.ckpt \
--output_graph=./mobilenet-freeze_and_quantize/frozen_graph/platerec_model_graph_10928f2_reshape.pb \
--output_node_names=ArgMax

sleep 1

python ./tools/optimize_for_inference.py \
--input=mobilenet-freeze_and_quantize/frozen_graph/platerec_model_graph_10928f2_reshape.pb \
--output=mobilenet-freeze_and_quantize/optimize_frozen_graph/platerec_model_graph_10928f2_opt_reshape.pb \
--frozen_graph=True \
--input_names=Placeholder \
--output_names=ArgMax

sleep 1

python ./tools/dzhang_quantize_graph.py \
--input=./mobilenet-freeze_and_quantize/optimize_frozen_graph/platerec_model_graph_10928f2_opt_reshape.pb \
--output_node_names="ArgMax" \
--output=./mobilenet-freeze_and_quantize/quantized_graph/quantized_platerec_model_graph_10928f2_opt_reshape.pb \
--mode=eightbit \
--parameter_file_name=/home/dzhang/work/quantized-mobilenet/statistical_requantization_range/tensorflow/32bit_to_8bit_parameter.txt

