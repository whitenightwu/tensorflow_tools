
# cd /home/ydwu/framework/tensorflow
# bazel run tensorflow/contrib/lite/tools:visualize -- /home/ydwu/tools/tf_tools/network/tf-lite/foo.tflite model_viz.html

cd /home/ydwu/framework/tensorflow/tensorflow/contrib/lite/tools/

python visualize.py /home/ydwu/tools/tf_tools/network/tf-lite/foo.tflite ./foo.html 

# --html_output
# --tflite_input
