
# ./tensorboard.sh /home/ydwu/project/facenet-fake/result_models

if [ ! $1 ]
then
    # DIR_FILE=/home/ydwu/project/mobilenet-quant-Q_DW/graph_transforms-Q_DW/result_model
    # DIR_FILE=/home/ydwu/project/facenet-fake/result_models
    DIR_FILE=/home/ydwu/project/facenet-quant/graph_transforms/result_model
else
    DIR_FILE=$1
fi

echo input_graph is \'${DIR_FILE}\'
   
echo "#####################################################################"
echo "####################### tensorboard #######################"
echo "#####################################################################"


CUDA_VISIBLE_DEVICES="" tensorboard --logdir=${DIR_FILE} --port=6007
