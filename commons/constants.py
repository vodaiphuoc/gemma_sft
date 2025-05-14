from typing import Literal

MODEL_KEY2IDS = {
    "bert": "google-bert/bert-base-uncased",
    "gemma": "google/gemma-3-1b-it"
}

LORA_PARAMS = {
    "r":16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias":"none",
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
}

DISTRIBUTION_TYPES = Literal["No","cuda","tpu"]

COLLATOR_INST_TEMPLATE = "<bos><start_of_turn>user"
COLLATOR_RESP_TEMPLATE = "<start_of_turn>model"