#!/bin/bash
# push local checkpoints to HF
subapp_dir=$(dirname "$(realpath "$0")")


echo $subapp_dir/checkpoints

OPTS=$(getopt -o "" --long hf: -- "$@")
eval set -- "$OPTS"

huggingface-cli login --token $2
# huggingface-cli upload [repo_id] [local_path] [path_in_repo]

huggingface-cli upload Daiphuoc/gemma3-loras $subapp_dir/checkpoints