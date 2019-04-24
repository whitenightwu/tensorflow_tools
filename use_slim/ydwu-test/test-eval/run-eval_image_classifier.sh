
# ./run-eval_image_classifier.sh > qqq1.log 2>&1

CHECKPOINT_FILE=/home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/scripts/mobilenet/tmp2/mobilenet_v1_1.0_224.ckpt

DATASET_DIR=/mllib/ImageNet/ILSVRC2012_tensorflow


CUDA_VISIBLE_DEVICES="" python ../../eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1 \
    --max_num_batches=10

# --batch_size=1000
# --master=/home/ydwu/framework/tensorflow22/tensorflow/ \
