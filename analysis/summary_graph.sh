
# PB_MODEL="/home/ydwu/project/fake-to-quant/network/fn-graph_transforms/quantized_graph.pb"

PB_MODEL="/home/ydwu/quant_tmp/DW-conv/result_model/quantized_graph.pb"

# PB_MODEL="/home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/scripts/mobilenet/tmp/mobilenet_v1_1.0_224/quantized_graph.pb"

# PB_MODEL=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/network/result/merge-BN.pb

# PB_MODEL=/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/merge-mean/tmp/merge-BN.pb


#######################

/home/ydwu/framework/tensorflow1.7/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
    --in_graph=${PB_MODEL} \
    --print_structure=true
