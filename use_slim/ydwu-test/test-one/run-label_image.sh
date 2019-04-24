
IMAGE_SIZE=224

MODEL_FOLDER=/home/ydwu/quant_tmp/DW-conv/result_model/quantized_graph.pb

LABEL_FOLDER=/home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/scripts/mobilenet/tmp/mobilenet_v1_1.0_224/labels.txt


echo "*******"
echo "Running label_image using the graph"
echo "*******"
# bazel build tensorflow/examples/label_image:label_image
/home/ydwu/framework/tensorflow1.7/bazel-bin/tensorflow/examples/label_image/label_image \
    --input_layer=input \
    --output_layer=MobilenetV1/Predictions/Reshape_1 \
    --graph=${MODEL_FOLDER} \
    --input_mean=-127 \
    --input_std=127 \
    --image=/home/ydwu/framework/tensorflow/tensorflow/examples/label_image/data/grace_hopper.jpg \
    --input_width=${IMAGE_SIZE} \
    --input_height=${IMAGE_SIZE} \
    --labels=${LABEL_FOLDER} \
    --self_test=true  


