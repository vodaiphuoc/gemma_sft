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
        print('self.is_fsdp_xla_v2_enabled: ', self.is_fsdp_xla_v2_enabled)
        if len(args) > 0:
            for _arg in args:
                print('arg: ',_arg, ', type: ', type(_arg))
        if len(kwargs) > 0:
            for _kwarg in kwargs:
                print('kwarg: ',_kwarg, ', type: ', type(_kwarg))
        
        model = super()._wrap_model(*args, **kwargs)
        print('result: ', model)

        use_accelerator_prepare = True if model is self.model else False
        print('check use_accelerator_prepare: ', use_accelerator_prepare)
        print('self.is_fsdp_enabled: ', self.is_fsdp_enabled)
        return model


    def torch_jit_model_eval(self, *args, **kwargs):
        print('model input to torch_jit_model_eval func: ', args[0], type(args[0]))
        return super().torch_jit_model_eval(*args, **kwargs)