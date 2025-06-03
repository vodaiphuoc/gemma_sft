#!/bin/bash
# do setup:
# - dependencies: ngrok, bitsandbytes
# - load lora from huggingface

OPTS=$(getopt -o "" --long lora_path: -- "$@")
eval set -- "$OPTS"

echo setup ngrok
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt -qq update \
  && sudo apt -qq install ngrok
echo done setup ngrok

echo setup bitsandbytes
pip install --quiet bitsandbytes
echo done isntall bitsandbytes

huggingface-cli download \
  Daiphuoc/gemma3-loras \
  --local-dir "$2"