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
            echo currently not support run bert with tpu
            exit 1

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
                    --model_key bart \
                    --train_batch_size 16 \
                    --eval_batch_size 16

            elif [ "${dist_type}" == "ddp" ]; then
                echo run train.py with ddp
                accelerate launch \
                    --config_file config/bart_gpus.yaml \
                    train.py \
                    --distribution_device cuda \
                    --distribution_type ddp \
                    --model_key bart \
                    --train_batch_size 8 \
                    --eval_batch_size 8
                
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
