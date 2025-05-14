# a mock trainer from trl for debugging
from trl import SFTTrainer
from typing import Union, Optional, Any
import warnings
import numpy as np
import torch

import copy

from accelerate.utils import (
        AutocastKwargs
)

class MockTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def torch_jit_model_eval(self, model, dataloader, training=False):
        print('running mock torch_jit_model_eval')
        if not training:
            if dataloader is None:
                print("failed to use PyTorch jit mode due to current dataloader is none.")
                return model
            example_batch = next(iter(dataloader))
            example_batch = self._prepare_inputs(example_batch)
            try:
                jit_model = copy.copy(model)
                jit_model.eval()
                original_forward = jit_model.__dict__.pop("_original_forward", None)
                # remove mixed precision hooks from the model
                if original_forward:
                    jit_model.forward = original_forward
                autocast_handler = AutocastKwargs(cache_enabled=False)
                with self.accelerator.autocast(autocast_handler=autocast_handler), torch.no_grad():
                    if isinstance(example_batch, dict):
                        jit_model = torch.jit.trace(jit_model, example_kwarg_inputs=example_batch, strict=False)
                        print('jit_model: ', jit_model)
                    else:
                        jit_model = torch.jit.trace(
                            jit_model,
                            example_kwarg_inputs={key: example_batch[key] for key in example_batch},
                            strict=False,
                        )
            except (RuntimeError, TypeError, ValueError, NameError, IndexError) as e:
                print(f"1) failed to use PyTorch jit mode due to: {e}.")

            try:
                jit_model = torch.jit.freeze(jit_model)
                with torch.no_grad():
                    jit_model(**example_batch)
                    jit_model(**example_batch)
                model = jit_model
                self.use_cpu_amp = False
            except (RuntimeError, TypeError, ValueError, NameError, IndexError) as e:
                print(f"2) failed to use PyTorch jit mode due to: {e}.")

        return model