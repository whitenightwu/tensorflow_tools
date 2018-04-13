
.././bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph="/home/ydwu/framework/tensorflow22/tensorflow/ydwu-quan_tools/merge-mean/tmp/merge-BN.pb" \
    --input_layer="input:0" \
    --input_layer_shape="1,224,224,3" \
    --input_layer_type="float" \
    --output_layer="MobilenetV1/Predictions/Reshape_1:0"

# bazel-bin/tensorflow/tools/benchmark/benchmark_model \
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
