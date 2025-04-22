#!/bin/bash

# trained by opt-1.3b reward model

MODEL_NAME="PKU-Alignment/beaver-7b-v3.0-reward"

CUDA_VISIBLE_DEVICES=6 python reward_eval.py \
    --data_path opt-1.3b_test_result.json \
    --model_name_or_path_reward $MODEL_NAME 
    
