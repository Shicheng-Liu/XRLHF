#!/bin/bash

set -x
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


DEV=5
PORT=1235
OUTPUT=$1
ZERO_STAGE=$2
MODEL_NAME="facebook/opt-1.3b"
DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/rlhf/dataset/Dahoas/full-hh-rlhf"

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt-1.3b/full-hh-rlhf
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

(deepspeed ---include localhost:$DEV --master_port $PORT \
main.py --model_name_or_path $MODEL_NAME \
   --num_padding_at_beginning 1 --weight_decay 0.1 --dropout 0.0 --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
   --data_path $DATA_PATH \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT) 2>&1 | tee $OUTPUT/training.log
