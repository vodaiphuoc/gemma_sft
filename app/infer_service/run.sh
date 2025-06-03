#!/bin/bash
app_dir=$(dirname $(dirname "$(realpath "$0")"))
lora_module_path=$app_dir/checkpoints
base_model="google/gemma-3-1b-it"
sup_lora_path="$lora_module_path/2025-05-30_13-40-23"
main_lora_path="$lora_module_path/2025-05-30_15-57-36"

lora_modules='{"name": "ftlora_sup", "path": '"$sup_lora_path"',"base_model_name": '"$base_model"'}'\
' {"name": "ftlora_main", "path": '"$main_lora_path"',"base_model_name": '"$base_model"'}'


echo $lora_modules

OPTS=$(getopt -o "" --long hf:,ngrok: -- "$@")
eval set -- "$OPTS"

setup_path=$app_dir/infer_service/setup.sh
bash "$setup_path" --lora_path "$lora_module_path"

huggingface-cli login --token $2

ngrok config add-authtoken $4
vllm serve "$base_model" \
    --dtype float16 \
    --task generate \
    --quantization bitsandbytes \
    --trust_remote_code \
    --max-model-len 1024 \
    --chat-template "$lora_module_path/2025-05-30_15-57-36/chat_template.jinja" \
    --enable-lora \
    --lora-modules "$lora_modules"
ngrok http http://0.0.0.0:8000 