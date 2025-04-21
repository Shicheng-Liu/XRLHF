#!/bin/bash

set -x
export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=20
export TOKENIZERS_PARALLELISM=False

DEV=0,3
PORT=1235
OUTPUT=$1
ZERO_STAGE=$2
MODEL_NAME="EleutherAI/pythia-2.8b"
DATA_PATH="/gpu02home/hbz5148/workspace/siyuan/rlhf/dataset/Dahoas/full-hh-rlhf"

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/pythia-2.8b/full-hh-rlhf
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

(deepspeed --include localhost:$DEV --master_port $PORT \
main.py \
   --model_name_or_path $MODEL_NAME \
   --gradient_accumulation_steps 8 \
   --lora_dim 128 \
   --zero_stage $ZERO_STAGE \
   --data_path $DATA_PATH \
   --enable_tensorboard \
   --tensorboard_path "$OUTPUT" \
   --output_dir "$OUTPUT" \
   --deepspeed) 2>&1 | tee "$OUTPUT/training.log"
