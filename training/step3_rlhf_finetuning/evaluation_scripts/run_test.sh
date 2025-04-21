#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
#~/workspace/siyuan/rlhf/training/step2_reward_model_finetuning/output/opt-350m/full-hh-rlhf 

MODEL_NAME="PKUAlignment/beaver-7b-v3.0-reward"

CUDA_VISIBLE_DEVICES=6 python test.py \
    --model_name_or_path_baseline facebook/opt-1.3b \
    --model_name_or_path_finetune ~/workspace/siyuan/rlhf/training/step1_supervised_finetuning/output/opt-1.3b/full-hh-rlhf \
    --model_name_or_path_rlhf output/opt-350m/full-hh-rlhf/actor \
    --data_path /gpu02home/hbz5148/workspace/siyuan/rlhf/dataset/Dahoas/full-hh-rlhf/test.json \
    --model_name_or_path_reward $MODEL_NAME 
    
