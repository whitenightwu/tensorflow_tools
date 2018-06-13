#!/bin/bash
## password: linux123

#######################################################################
# # # 193.169.1.229

# # ps 
# ssh ydwu@193.169.1.230 "export PATH=~/anaconda2/bin:$PATH; \
#              cd tmp/distributeTensorflowExample; \
#              source activate tf_1.8-ydwu; \
# CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=ps --task_index=0" \
# ; \
# # worker0
# ssh ydwu@193.169.1.230 "export PATH=~/anaconda2/bin:$PATH; \
#              cd tmp/distributeTensorflowExample; \
#              source activate tf_1.8-ydwu; \
# CUDA_VISIBLE_DEVICES='0' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=worker --task_index=0" \

# ; \
# # worker1
# CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=worker --task_index=1  



#######################################################################
# # 193.169.1.230

# # ps 
# CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=ps --task_index=0


# # worker0 & worker1
CUDA_VISIBLE_DEVICES='0' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=worker --task_index=0 \
& ssh ydwu@193.169.1.229 "export PATH=~/anaconda2/bin:$PATH; \
             cd tmp/distributeTensorflowExample; \
             source activate 229_tf_1.8; \
CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=worker --task_index=1"
