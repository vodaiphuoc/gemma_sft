from .chat_template import adjust_tokenizer
from .constants import DISTRIBUTION_TYPES, MODEL_KEY2IDS, LORA_PARAMS

from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig
)
import torch
from typing import Tuple, Union
from types import NoneType


def _get_pretrained_model(
        model_id:str, 
        distribution_type: DISTRIBUTION_TYPES
    )->PreTrainedModel:
    r"""
    Get pretrained model depend on `distribution_type`:
        - in distribution on TPU, no device map used
        - in distribution on GPUs, map device with `PartialState`
    """
    if distribution_type == "cuda":
        from accelerate import PartialState
        device_map={'':PartialState().process_index}
    elif distribution_type == "tpu":
        device_map=None
    else:
        device_map=None
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='eager',
        torch_dtype=torch.bfloat16,
        # device_map="cpu",
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16
        )
    )
    return model

def get_model_tokenizer(
        model_key:str = "gemma",
        distribution_type: DISTRIBUTION_TYPES = "cuda"
    )->Tuple[PreTrainedModel, PreTrainedTokenizer,Union[LoraConfig, NoneType]]:
    
    if model_key == "gemma":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY2IDS[model_key])
        model = _get_pretrained_model(
            model_id= MODEL_KEY2IDS[model_key],
            distribution_type = distribution_type
        )
        lora_config = LoraConfig(
            **LORA_PARAMS
        )
        return model, tokenizer, lora_config

    elif model_key == "bert":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY2IDS[model_key])
        model = _get_pretrained_model(
            model_id= MODEL_KEY2IDS[model_key],
            distribution_type = distribution_type
        )
        model, tokenizer = adjust_tokenizer(model, tokenizer)
        return model, tokenizer, None
        
    else:
        raise NotImplemented(f"Only support key in: {list(MODEL_KEY2IDS.keys())}")