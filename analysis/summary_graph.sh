

if [ ! $1 ]
then
    # PB_MODEL="/home/ydwu/project/mobilenet-quant-Q_DW/graph_transforms-Q_DW/result_model/quantized_graph.pb"
    # PB_MODEL="/home/ydwu/project/mobilenet-quant-Q_DW/quantization-Q_DW/result_model/quantization.pb"
    PB_MODEL="/home/ydwu/project/facenet-quant/graph_transforms/result_model/quantized_graph.pb"
else
    PB_MODEL=$1
fi

echo input_graph is \'${PB_MODEL}\'
   
echo "#####################################################################"
echo "####################### summarize_graph #######################"
echo "#####################################################################"

/home/ydwu/framework/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
    --in_graph=${PB_MODEL} \
    --print_structure=true

echo "summary_graph.sh === complete!!!"

