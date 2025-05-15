# a mock trainer from trl for debugging

from typing import Union, Optional, Any
import warnings
import numpy as np
import torch

from trl import SFTTrainer


class MockSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _wrap_model(self, *args, **kwargs):
        if len(args) > 0:
            for _arg in args:
                print('arg: ',_arg, ', type: ', type(_arg))
        if len(kwargs) > 0:
            for _kwarg in kwargs:
                print('kwarg: ',_kwarg, ', type: ', type(_kwarg))
        
        result = super()._wrap_model(*args, **kwargs)
        print('result: ', result)

        use_accelerator_prepare = True if model is self.model else False
        print('check use_accelerator_prepare: ', use_accelerator_prepare)
        return result