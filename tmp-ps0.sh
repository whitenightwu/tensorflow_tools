#!/bin/bash
## linux123


ssh ydwu@193.169.1.230 "export PATH=~/anaconda2/bin:$PATH; \
             cd tmp/distributeTensorflowExample; \
             source activate tf_1.8-ydwu; \
CUDA_VISIBLE_DEVICES='0' python distribute.py --ps_hosts=193.169.1.230:6666 --worker_hosts=193.169.1.230:7777,193.169.1.229:8888 --job_name=worker --task_index=0"
