#!/bin/bash
# ./train_all_data_at_once Model Custom_Shared_Args

MODEL=${1:-'GCAKE'}
LOGDIR=log/${MODEL}_All_`date +"%m_%d-%H_%M_%S"`
PROCESS_NAME="HelloWorld"

CUSTOM_ARGS=${2:-''}
SHARED_ARGS="--model $MODEL --logdir $LOGDIR $CUSTOM_ARGS --process-name $PROCESS_NAME"

mkdir -p $LOGDIR

select_gpu () {
    # get top K GPUs which has max free memory
    K=$1
    echo `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -rn -k1 | head -$K | awk '{print $1}' ORS=''`
}

# Use multiple GPUs
GPU=`select_gpu 2`
export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU: $GPU"


# FB15K-237
nohup python3 manage.py --dataset FB15K-237 $SHARED_ARGS > /dev/null 2> $LOGDIR/FB15K-237_Error.log &

# WN18RR
nohup python3 manage.py --dataset WN18RR $SHARED_ARGS > /dev/null 2> $LOGDIR/WN18RR_Error.log &

echo "== Running scripts =="
ps aux | grep $PROCESS_NAME
echo "To stop all processes execute:"
echo "kill -9" `ps x | grep $PROCESS_NAME | grep -v grep | awk '{print $1}' | xargs`

echo "== Tensorboard =="
echo "Tensorboard should be run at port 6006"
echo "To terminate it just press CTRL+C to quit"
tensorboard --logdir $LOGDIR/.
