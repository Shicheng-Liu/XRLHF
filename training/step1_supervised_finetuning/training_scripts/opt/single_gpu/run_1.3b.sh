#!/bin/bash

set -x
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTPUT=$1
ZERO_STAGE=$2
MODEL_NAME="facebook/opt-1.3b"
DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/rlhf/dataset/Dahoas/rm-static"

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT


deepspeed --num_gpus 1 main.py --model_name_or_path $MODEL_NAME \
   --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
   --data_path $DATA_PATH
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT 2>&1 | tee $OUTPUT/training.log
