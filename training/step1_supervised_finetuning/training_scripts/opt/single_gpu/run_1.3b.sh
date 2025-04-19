#!/bin/bash
<<<<<<< HEAD
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
=======

set -e

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


DATA_PATH="~/workspace/siyuan/rlhf/dataset/Dahoas/rm-static"
MODEL_NAME="facebook/opt-1.3b"


>>>>>>> 25b97a1 (commit)
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

<<<<<<< HEAD
deepspeed --num_gpus 1 main.py --model_name_or_path pretrain \
   --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
=======
deepspeed --num_gpus 1 main.py --model_name_or_path $MODEL_NAME \
   --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
   --data_path $DATA_PATH
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT 2>&1 | tee $OUTPUT/training.log
>>>>>>> 25b97a1 (commit)
