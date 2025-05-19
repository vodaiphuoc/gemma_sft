#!/bin/bash

OPTS=$(getopt -o "" --long tpu:,type: -- "$@")
eval set -- "$OPTS"

run_with_tpu=""
dist_type=""

case "$1" in
    --tpu)
        run_with_tpu="$2"
        dist_type="$4"

        if [ "${run_with_tpu}" == "true" ]; then
            # with TPU, use notebook lunch + latest version (1.7.0.dev0) and
            # dont use accelerate launch
            echo run with tpu

            echo setup sub dependencies in requirement
            pip install -q -U -r dependencies/tpu_train_requirements.txt
            pip install -q git+https://github.com/huggingface/accelerate
            
            echo run train.py
            python train.py \
                --distribution_device tpu \
                --model_key gemma \
                --fsdp_config_path config/gemma_tpu.yaml

        elif [ "${run_with_tpu}" == "false" ]; then
            # with GPUs, use accelerate lunch with 1.6.0
            echo setup sub dependencies in requirement for gpu
            pip install -q -U -r dependencies/train_requirements.txt
            
            # echo check version after install
            # pip list

            
            if [ "${dist_type}" == "fsdp" ]; then
                echo run train.py with fsdp
                accelerate launch \
                    --config_file config/gemma_fsdp.yaml \
                    train.py \
                    --distribution_device cuda \
                    --distribution_type fsdp \
                    --model_key gemma \
                    --train_batch_size 2 \
                    --eval_batch_size 2

            elif [ "${dist_type}" == "ddp" ]; then
                echo run train.py with ddp
                accelerate launch \
                    --config_file config/gemma_gpus.yaml \
                    train.py \
                    --distribution_device cuda \
                    --distribution_type ddp \
                    --model_key gemma \
                    --train_batch_size 6 \
                    --eval_batch_size 6
                
            else
                echo only support distribution type "fsdp" or "ddp"
                exit 1
            fi

        else
            echo unknow value of tpu option: "${run_with_tpu}"
            exit 1

        fi
    ;;
    *)
        exit 1
    ;;
esac
