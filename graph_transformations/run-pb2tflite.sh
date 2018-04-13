


# bazel build tensorflow/contrib/lite/toco:toco

/home/ydwu/framework/tensorflow/bazel-bin/tensorflow/contrib/lite/toco/toco \
    --input_file=/home/ydwu/framework/tensorflow/ydwu-quan-2/tmp/frozen_eval_graph.pb \
    --output_file=/home/ydwu/framework/tensorflow/ydwu-quan-2/tmp/ttt.dot \
    --input_format=TENSORFLOW_GRAPHDEF \
    --output_format=GRAPHVIZ_DOT \
    --inference_type=FLOAT \
    --input_shape="1,224,224,3" \
    --input_array=input \
    --output_array=MobilenetV1/Predictions/Reshape_1 \
    --std_value=127.5 \
    --mean_value=127.5 \
    --logtostderr \
    --v=1

# inference_type=
# QUANTIZED_UINT8 
# FLOAT

# output_format=
# TFLITE  foo.tflite
# GRAPHVIZ_DOT  ttt.dot
