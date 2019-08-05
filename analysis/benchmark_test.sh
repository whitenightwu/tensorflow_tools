
# ./bazel-bin/tensorflow/tools/benchmark/benchmark_model \
# --graph="/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/merge-mean/tmp/merge-BN.pb" \
# --input_layer="input:0" \
# --input_layer_shape="1,224,224,3" \
# --input_layer_type="float" \
# --output_layer="MobilenetV1/Predictions/Reshape_1:0" \
# --show_run_order=false \
# --show_time=false \
# --show_memory=false   \
# --show_summary=true   \
# --show_flops=true   \
# --logtostderr


# ./bazel-bin/tensorflow/tools/benchmark/benchmark_model \
#     --graph="/home/ydwu/datasets/squeezenet-20180204-160909/frozen_eval_graph.pb" \
#     --input_layer="Placeholder:0" \
#     --input_layer_shape="1,160,160,3" \
#     --input_layer_type="float" \
#     --output_layer="embeddings:0"


# ./bazel-bin/tensorflow/tools/benchmark/benchmark_model \
#     --graph="/home/ydwu/datasets/mtcnn/mtcnn_freezed_model.pb" \
#     --input_layer="rnet/input:0" \
#     --input_layer_shape="1,24,24,3" \
#     --input_layer_type="float" \
#     --output_layer="rnet/prob1:0"
