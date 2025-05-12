from .chat_template import adjust_tokenizer
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)

from typing import Tuple, Union, Literal
from types import NoneType

MODEL_KEY2IDS = {
    "bert": "google-bert/bert-base-uncased",
    "gemma": "google/gemma-3-1b-it",
    "gemma_unsloth": "unsloth/gemma-3-1b-it"
}

LORA_PARAMS = {
    "r":16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
}

DISTRIBUTION_TYPES = Literal["No","cuda","tpu"]

def _get_pretrained_model(
        model_id:str, 
        distribution_type: DISTRIBUTION_TYPES
    )->PreTrainedModel:
    if distribution_type == "cuda":
        from accelerate import PartialState
        device_map={'':PartialState().process_index}
    elif distribution_type == "tpu":
        import torch_xla.core.xla_model as xm        
        device_map={'':xm.xla_device()}
    elif distribution_type == "No":
        device_map=None
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='eager',
        device_map=device_map
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

    elif model_key == "gemma_unsloth":
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name= MODEL_KEY2IDS[model_key],
        )
        model = FastLanguageModel.get_peft_model(model,**LORA_PARAMS)
        return model, tokenizer, None

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