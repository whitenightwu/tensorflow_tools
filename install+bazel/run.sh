
#bazel clean

####
pip uninstall tensorflow 
sleep 1

####
bazel build --config=opt  //tensorflow/tools/pip_package:build_pip_package
# bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
# bazel build -c opt --jobs 1 //tensorflow/cc:tutorials_example_trainer 
sleep 1

####
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sleep 1

####
pip install /tmp/tensorflow_pkg/tensorflow-XXXXXX
# pip install /tmp/tensorflow_pkg/tensorflow-1.5.0rc1-cp27-cp27mu-linux_x86_64.whl 
# pip install /tmp/tensorflow_pkg/tensorflow-1.6.0-cp27-cp27mu-linux_x86_64.whl
sleep 1

####
cd /home/ydwu/framework/tensorflow22/tensorflow/tensorflow-models/research/slim/ydwu-test
sleep 1

./run-eval_image_classifier.sh 
sleep 1


# /usr/include/x86_64-linux-gnu/cudnn_v7.h
