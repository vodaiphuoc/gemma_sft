#!/bin/bash

OPTS=$(getopt -o "" --long hf:,ngrok: -- "$@")
eval set -- "$OPTS"

hf_token="$2"
ngrok_token="$4"

ngrok config add-authtoken $ngrok_token
ngrok http http://0.0.0.0:8000

vllm serve google/gemma-3-1b-it \
    --dtype float16 \
    --task generate \
    --trust_remote_code \
    --max-model-len 128 \
    --enable-lora \
    --lora-modules ftlora=/kaggle/working/gemma_sft/app/checkpoints/2025-05-30_13-40-23