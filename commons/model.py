from .bart_utils import adjust_tokenizer, extend_position_embedding
from .constants import (
    DISTRIBUTION_TYPE, 
    DISTRIBUTION_DEVICE,
    MODEL_KEY2IDS, 
    LORA_PARAMS1,
    LORA_PARAMS2,
    BART_LORA_PARAMS
)

from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)
import torch
from typing import Tuple, Union
from types import NoneType
import os

def _get_pretrained_model(
        model_id:str,
        distribution_device: DISTRIBUTION_DEVICE,
        distribution_type: DISTRIBUTION_TYPE
    )->PreTrainedModel:
    r"""
    Get pretrained model depend on `distribution_type`:
        - in distribution on TPU, no device map used
        - in distribution on GPUs, map device with `PartialState`
    """
    if distribution_device == "cuda":
        if distribution_type == "fsdp":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16
            )
            loading_dtype = torch.bfloat16
        
        elif distribution_type == "ddp":
            quantization_config=None
            loading_dtype = torch.float32
        
        else:
            raise NotImplementedError

    elif distribution_device == "tpu":
        quantization_config=None
        loading_dtype = None
    else:
        quantization_config=None
        loading_dtype = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='eager',
        torch_dtype=loading_dtype,
        quantization_config = quantization_config,
    )
    return model

def get_model_tokenizer(
        model_key:str = "gemma",
        distribution_device: DISTRIBUTION_DEVICE = "cuda",
        distribution_type: DISTRIBUTION_TYPE = "ddp",
        checkpoint_dir: str = None
    )->Tuple[PreTrainedModel, PreTrainedTokenizer,Union[LoraConfig, NoneType]]:
    
    if model_key == "gemma":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY2IDS[model_key])

        if checkpoint_dir is None:
            model = _get_pretrained_model(
                model_id= MODEL_KEY2IDS[model_key],
                distribution_device = distribution_device,
                distribution_type = distribution_type
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

        lora_config = LoraConfig(
            **LORA_PARAMS1
        )
        return model, tokenizer, lora_config

    elif model_key == "bart":
        tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__).replace("commons","tokenizer"))

        if checkpoint_dir is None:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_KEY2IDS[model_key],
                attn_implementation='eager',
                torch_dtype=torch.float32
            )
            model = extend_position_embedding(model, new_context_length= 2048)
            
        else:
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

        model, tokenizer = adjust_tokenizer(model, tokenizer)
        
        lora_config = LoraConfig(
            **BART_LORA_PARAMS
        )
        return model, tokenizer, None
        
    else:
        raise NotImplemented(f"Only support key in: {list(MODEL_KEY2IDS.keys())}")