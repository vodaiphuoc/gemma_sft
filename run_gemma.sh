#!/bin/bash

OPTS=$(getopt -o "" --long tpu:: -- "$@")
eval set -- "$OPTS"

run_with_tpu=""
case "$1" in
    --tpu)
        shift 2
        run_with_tpu="$2"

        if [ "${run_with_tpu}" == "true" ]; then
            echo run with tpu

            # echo setup optimum-tpu for tpu
            # pip install optimum-tpu -qf https://storage.googleapis.com/libtpu-releases/index.html
            
            echo setup dependencies in requirement
            # pip install -q -U -r dependencies/tpu_train_requirements.txt
            pip install -q peft
            echo run train_gemma.py
            accelerate launch \
                --config_file config/gemma_tpu.yaml \
                train_gemma.py \
                --distribution_type tpu \
                --model_key gemma

        elif [ "${run_with_tpu}" == "false" ]; then
            echo setup dependencies in requirement

            pip install -q -U -r dependencies/train_requirements.txt
            pip list
            
            echo run with gpu
            accelerate launch \
                --config_file config/gemma.yaml \
                train_gemma.py \
                --distribution_type cuda \
                --model_key gemma
        
        else
            echo unknow value of tpu option: "${run_with_tpu}"
            exit 1

        fi
    ;;
    *)
        exit 1
    ;;
esac
