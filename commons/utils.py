import yaml
import json
import os
from typing import Dict, Union
from types import NoneType
from transformers import TrainerCallback
import matplotlib.pyplot as plt
from torch.optim import Optimizer

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


class LearningRateLogger(TrainerCallback):
    r"""
    Custom logger for collecting learing rate
    """
    def __init__(self):
        self._lr_per_step = []

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

    def on_train_end(self, args, state, control, **kwargs):
        try:
            logging_file_path = os.path.join(args.logging_dir, "learning_rate.json")
            with open(logging_file_path) as fp:
                json.dump(self._lr_per_step, fp)
        except Exception as e:
            print(f"error in on_train_end callback: {e}")
        
        finally:
            plt.plot(
                [int(ele['step']) for ele in self._lr_per_step],
                [ele['lr'] for ele in self._lr_per_step]
            )
            plt.savefig(os.path.join(args.logging_dir, "lr_plot.png"))