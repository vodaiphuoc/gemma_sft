#!/bin/bash

OPTS=$(getopt -o "" --long hf:,ngrok: -- "$@")
eval set -- "$OPTS"

echo setup ngrok
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt -qq update \
  && sudo apt -qq install ngrok
echo done setup ngrok

huggingface-cli login --token $2

ngrok config add-authtoken $4
ngrok http http://0.0.0.0:8000 & vllm serve google/gemma-3-1b-it \
    --dtype float16 \
    --task generate \
    --trust_remote_code \
    --max-model-len 512 \
    --enable-lora \
    --lora-modules ftlora=/kaggle/working/gemma_sft/app/checkpoints/2025-05-30_13-40-23