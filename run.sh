# # # the Cluster
# CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=193.169.1.231:6666 --worker_hosts=193.169.1.231:7777 --job_name=ps --task_index=0  

# # # the Client
# CUDA_VISIBLE_DEVICES='0' python distribute.py --ps_hosts=193.169.1.231:6666 --worker_hosts=193.169.1.231:7777 --job_name=worker --task_index=0  


# #
# CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=193.169.1.231:6666 --worker_hosts=193.169.1.231:7777,193.169.1.229:8888 --job_name=ps --task_index=0  
# CUDA_VISIBLE_DEVICES='0' python distribute.py --ps_hosts=193.169.1.231:6666 --worker_hosts=193.169.1.231:7777,193.169.1.229:8888 --job_name=worker --task_index=0  
# CUDA_VISIBLE_DEVICES='0' python distribute.py --ps_hosts=193.169.1.231:6666 --worker_hosts=193.169.1.231:7777,193.169.1.229:8888 --job_name=worker --task_index=1  



CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=ps --task_index=0
CUDA_VISIBLE_DEVICES='0' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=worker --task_index=0
CUDA_VISIBLE_DEVICES='0' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=worker --task_index=1  
