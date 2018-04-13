

CHECKPOINT_FILE_GRAPH=/home/ydwu/framework/tensorflow/ydwu-quan-2/network/step-01/mobilenet_v1_1.0_224_quant_eval.pbtxt

CHECKPOINT=/home/ydwu/framework/tensorflow/ydwu-quan-2/network/step-01/mobilenet_v1_1.0_224_quant.ckpt

RESULT_FILE=/home/ydwu/framework/tensorflow/ydwu-quan-2/tmp/frozen_eval_graph.pb

OUTPUT=MobilenetV1/Predictions/Reshape_1

CUDA_VISIBLE_DEVICES="" python /home/ydwu/framework/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=${CHECKPOINT_FILE_GRAPH} \
  --input_checkpoint=${CHECKPOINT} \
  --output_graph=${RESULT_FILE} \
  --output_node_names=${OUTPUT}

