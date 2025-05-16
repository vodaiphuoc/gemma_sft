# a mock trainer from trl for debugging

from typing import Union, Optional, Any
import warnings
import numpy as np
import torch

from trl import SFTTrainer, SFTConfig, pack_dataset

class MockSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _prepare_dataset(self, *args, **kwargs):
        dataset = super()._prepare_dataset(*args, **kwargs)
        # get sft config
        config = None
        for _arg in args:
            if isinstance(_arg, SFTConfig):
                config = _arg
        for _, _kwarg_value in kwargs.items():
            if isinstance(_kwarg_value, SFTConfig):
                config = _kwarg_value
        
        example = dataset[0]
        assert len(example['input_ids']) == len(example['completion_mask'])

        map_kwargs = {
            "desc": f"Custom packing dataset in `MockSFTTrainer`"
        }
        packed_dataset = pack_dataset(dataset, config.max_length, map_kwargs)
        return packed_dataset