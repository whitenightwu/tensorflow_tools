CUDA_VISIBLE_DEVICES="" python xxx.py


from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '99'
# os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT']= '2'
# print("xxxxxxxxxxxxxxxxxxxxxxx = ", device_lib.list_local_devices())



# tf.device(['/gpu:2','/gpu:1'])
# print("yyyyyyyyyyyyyyyyyyyyy = ", tf.test.is_gpu_available())
