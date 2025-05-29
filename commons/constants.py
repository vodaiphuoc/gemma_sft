from typing import Literal

MODEL_KEY2IDS = {
    "bart": "facebook/bart-large",
    "gemma": "google/gemma-3-1b-it",
    "lstm": "google/gemma-3-1b-it"
}

LR_KEY2IDS = {
    "bart": {
        "init_lr": 1e-5,
        "min_lr": 1e-9,
        "num_cycles": 0.5
    },
    "gemma": {
        "init_lr": 1e-5,
        "min_lr": 1e-9,
        "num_cycles": 0.5
    },
    "lstm": {
        "init_lr": 1e-3,
        "min_lr": 1e-5,
        "num_cycles": 0.5
    }
}

LORA_PARAMS1 = {
    "r":8,
    "lora_alpha": 512,
    "lora_dropout": 0.05,
    "bias":"none",
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "down_proj","up_proj"],
    "task_type": "CAUSAL_LM"
}


LORA_PARAMS2 = {
    "r":16,
    "lora_alpha": 32,
    "lora_dropout": 0.02,
    "bias":"none",
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj"],
    "task_type": "CAUSAL_LM"
}


DISTRIBUTION_DEVICE = Literal["No","cuda","tpu"]
DISTRIBUTION_TYPE = Literal["fsdp","ddp"]

COLLATOR_INST_TEMPLATE = "<start_of_turn>user"
COLLATOR_RESP_TEMPLATE = "<start_of_turn>model"