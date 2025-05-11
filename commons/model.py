from .chat_template import adjust_tokenizer
from peft import LoraConfig
from accelerate import PartialState
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)
from unsloth import FastLanguageModel

from typing import Tuple, Union
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

def get_model_tokenizer(
        model_key:str = "gemma"
    )->Tuple[PreTrainedModel, PreTrainedTokenizer,Union[LoraConfig, NoneType]]:
    
    if model_key == "gemma":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY2IDS[model_key])
        device_string = PartialState().process_index
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_KEY2IDS[model_key],
            attn_implementation='eager',
            device_map={'':device_string}
        )
        lora_config = LoraConfig(
            **LORA_PARAMS
        )
        return model, tokenizer, lora_config

    elif model_key == "gemma_unsloth":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name= MODEL_KEY2IDS[model_key],
            # max_seq_length=max_length,
        )
        model = FastLanguageModel.get_peft_model(model,**LORA_PARAMS)
        return model, tokenizer, None

    elif model_key == "bert":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_KEY2IDS[model_key])
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_KEY2IDS[model_key]
        )
        model, tokenizer = adjust_tokenizer(model, tokenizer)
        return model, tokenizer, None
        
    else:
        raise NotImplemented(f"Only support key in: {list(MODEL_KEY2IDS.keys())}")