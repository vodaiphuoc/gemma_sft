import yaml
import json
import os
import torch
from typing import Dict, Union, Optional
from types import NoneType
from transformers import TrainerCallback
import matplotlib.pyplot as plt
from torch.optim import Optimizer
import numpy as np

def get_fsdp_config_from_yaml(yaml_path: str)->Union[Dict[str,str], NoneType]:
    if yaml_path != "":
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
            return data['fsdp_config']
        
        except Exception as e:
            print("error in open file yaml config: ",e)
        
    else:
        return None


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        pad_to_multiple_of (`int`, *optional*, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output



class LearningRateLogger(TrainerCallback):
    r"""
    Custom logger for collecting learing rate
    """
    def __init__(self):
        self._lr_per_step = []
        self._loss_per_step = []

    def on_step_end(self, args, state, control, **kwargs):
        current_lr = None
        for _kw, _kw_value in kwargs.items():
            if _kw == "optimizer" and isinstance(_kw_value, Optimizer):
                current_lr = _kw_value.param_groups[0]['lr']
                break
        
        if current_lr is not None:
            self._lr_per_step.append({
                "step": state.global_step,
                "lr": current_lr
            })

    def on_log(self, args, state, control, logs, **kwargs):
        if logs.get("loss") is not None:
            self._loss_per_step.append({
                "step": state.global_step,
                "loss": logs.get("loss")
            })

    def on_train_end(self, args, state, control, **kwargs):
        try:
            logging_file_path = os.path.join(args.logging_dir, "learning_rate.json")
            with open(logging_file_path,'w') as fp:
                json.dump(self._lr_per_step, fp)
        except Exception as e:
            print(f"error in on_train_end callback: {e}")
        
        finally:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[16, 8])

            ax1.plot(
                [int(ele['step']) for ele in self._lr_per_step],
                [ele['lr'] for ele in self._lr_per_step],
                label = 'learning rate'
            )
            ax1.legend()

            self._loss_per_step = [
                ele for ele in \
                self._loss_per_step \
                if ele['loss'] < 1.0
            ]

            ax2.plot(
                [int(ele['step']) for ele in self._loss_per_step],
                [ele['loss'] for ele in self._loss_per_step],
                label = 'training loss'
            )
            ax2.legend()
            plt.savefig(os.path.join(args.logging_dir, "lr_loss_plot.png"))