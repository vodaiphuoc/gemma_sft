#!/bin/bash

OPTS=$(getopt -o "" --long tpu:: -- "$@")
eval set -- "$OPTS"

run_with_tpu=""
case "$1" in
    --tpu)
        shift 2
        run_with_tpu="$2"

        if [ "${run_with_tpu}" == "true" ]; then
            # with TPU, use notebook lunch + latest version (1.7.0.dev0) and
            # dont use accelerate launch
            echo run with tpu

            echo setup sub dependencies in requirement
            pip install -q -U -r dependencies/tpu_train_requirements.txt
            pip install -q git+https://github.com/huggingface/accelerate
            
            echo run train.py
            python train.py \
                --distribution_type tpu \
                --model_key gemma \
                --fsdp_config_path config/gemma_tpu.yaml

        elif [ "${run_with_tpu}" == "false" ]; then
            # with GPUs, use accelerate lunch with 1.6.0
            echo run with gpu

            echo setup sub dependencies in requirement
            pip install -q -U -r dependencies/train_requirements.txt
            
            echo check version after install
            pip list

            echo run train.py
            accelerate launch \
                --config_file config/gemma.yaml \
                train.py \
                --distribution_type cuda \
                --model_key gemma \
                --train_batch_size 12 \
                --eval_batch_size 12
        
        else
            echo unknow value of tpu option: "${run_with_tpu}"
            exit 1

        fi
    ;;
    *)
        exit 1
    ;;
esac
