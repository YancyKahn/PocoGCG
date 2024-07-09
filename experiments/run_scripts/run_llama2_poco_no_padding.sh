#!/bin/bash

python_script_path="../../experiments_evaluate.py"
device="cuda:5"
model_name="llama-2"

python $python_script_path \
        --model_name $model_name  \
        --baseline "poco" \
        --padding_token "" \
        --batch_size 512 \
        --num_steps 200 \
        --topk 256 \
        --device $device