
DATASET_DIR=/mllib/ImageNet/ILSVRC2012_tensorflow
TRAIN_DIR=/home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/ydwu-train/fineturn
CHECKPOINT_PATH=/home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/scripts/mobilenet/tmp/mobilenet_v1_1.0_224.ckpt

CUDA_VISIBLE_DEVICES="" python ../train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=mobilenet_v1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_interval_secs=1 \
    --clone_on_cpu=true \
    --learning_rate=0.00000001 \
    --max_number_of_steps=1


    # --trainable_scopes=MobilenetV1/Conv2d_5_pointwise/BatchNorm/moving_mean
