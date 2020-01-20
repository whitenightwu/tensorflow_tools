
# ./tensorboard.sh /home/ydwu/project/facenet-fake/result_models

if [ ! $1 ]
then
    DIR_FILE=/tmp/tf-ydwu
    
else
    DIR_FILE=$1
fi

echo input_graph is \'${DIR_FILE}\'
   
echo "#####################################################################"
echo "####################### tensorboard #######################"
echo "#####################################################################"


CUDA_VISIBLE_DEVICES="" tensorboard --logdir=${DIR_FILE} --port=6007
