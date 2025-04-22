#!/bin/bash

# trained by opt-1.3b reward model

MODEL_NAME="PKU-Alignment/beaver-7b-v3.0-reward"

CUDA_VISIBLE_DEVICES=6 python test_batch.py \
    --reward_model_name opt-1.3b \
    --model_name_or_path_baseline facebook/opt-1.3b \
    --model_name_or_path_finetune ~/workspace/siyuan/rlhf/training/step1_supervised_finetuning/output/opt-1.3b/full-hh-rlhf \
    --model_name_or_path_rlhf output/opt-1.3b/full-hh-rlhf/actor \
    --data_path /gpuhome/hbz5148/workspace/siyuan/rlhf/dataset/Dahoas/full-hh-rlhf/test.json \
    --batch_size 8 \
    --model_name_or_path_reward $MODEL_NAME 
    
