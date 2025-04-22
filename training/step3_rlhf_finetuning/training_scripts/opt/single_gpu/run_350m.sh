#!/bin/bash

set -x
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=20
export TOKENIZERS_PARALLELISM=False

DEV=5,6
PORT=1236
BASELINE_ACTOR_MODEL_PATH=facebook/opt-1.3b
ACTOR_MODEL_PATH=~/workspace/siyuan/rlhf/training/step1_supervised_finetuning/output/opt-1.3b/full-hh-rlhf
CRITIC_MODEL_PATH=~/workspace/siyuan/rlhf/training/step2_reward_model_finetuning/output/opt-350m/full-hh-rlhf
DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/rlhf/dataset/Dahoas/full-hh-rlhf"


ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt-350m/full-hh-rlhf
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=0
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=0
fi
mkdir -p $OUTPUT

(deepspeed --include localhost:$DEV --master_port $PORT \
main.py \
   --baseline_actor_model_name_or_path $BASELINE_ACTOR_MODEL_PATH --actor_model_name_or_path $ACTOR_MODEL_PATH --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE --critic_zero_stage $CRITIC_ZERO_STAGE \
   --num_padding_at_beginning 1 --gradient_accumulation_steps 2 \
   --deepspeed --actor_lora_dim 128 --actor_gradient_checkpointing --actor_dropout 0.0 \
   --data_path $DATA_PATH \
   --deepspeed --output_dir $OUTPUT) 2>&1 | tee $OUTPUT/training.log
