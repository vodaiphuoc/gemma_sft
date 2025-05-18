from typing import Literal

MODEL_KEY2IDS = {
    "bert": "google-bert/bert-base-uncased",
    "gemma": "google/gemma-3-1b-it"
}

LORA_PARAMS1 = {
    "r":8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias":"none",
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
}


LORA_PARAMS2 = {
    "r":8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias":"none",
    "target_modules": ["all-linear"],
    "task_type": "CAUSAL_LM"
}


DISTRIBUTION_DEVICE = Literal["No","cuda","tpu"]
DISTRIBUTION_TYPE = Literal["fsdp","ddp"]

COLLATOR_INST_TEMPLATE = "<start_of_turn>user"
COLLATOR_RESP_TEMPLATE = "<start_of_turn>model"