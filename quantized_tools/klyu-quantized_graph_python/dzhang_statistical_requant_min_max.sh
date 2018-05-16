#!/bin/bash

python ./freeze_graph_preprocess.py
sleep 1
echo ydwu == ok-1

python ./tools/freeze_graph.py \
--input_graph=./mobilenet-freeze_and_quantize/ckpt_graph/platerec_model_graph_10928f2_reshape.pbtxt \
--input_checkpoint=./mobilenet-freeze_and_quantize/ckpt_graph/platerec_model_graph_10928f2_reshape.ckpt \
--output_graph=./mobilenet-freeze_and_quantize/frozen_graph/platerec_model_graph_10928f2_reshape.pb \
--output_node_names=ArgMax

sleep 1
echo ydwu == ok-2

python ./tools/optimize_for_inference.py \
--input=mobilenet-freeze_and_quantize/frozen_graph/platerec_model_graph_10928f2_reshape.pb \
--output=mobilenet-freeze_and_quantize/optimize_frozen_graph/platerec_model_graph_10928f2_opt_reshape.pb \
--frozen_graph=True \
--input_names=Placeholder \
--output_names=ArgMax

sleep 1
echo ydwu == ok-3

python ./tools/dzhang_statistical_requant_min_max.py \
--input=./mobilenet-freeze_and_quantize/optimize_frozen_graph/platerec_model_graph_10928f2_opt_reshape.pb \
--output_node_names="ArgMax" \
--output=./mobilenet-freeze_and_quantize/quantized_graph/quantized_platerec_model_graph_10928f2_opt_for_statistical_min_max.pb \
--mode=eightbit

echo ydwu == ok-end
