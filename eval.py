from commons.inference import Serving
from commons.dataset import get_datasets
import torch
import os

current_time = "2025-05-20_22-42-32"

_, _, testdataset = get_datasets(version="3_1", ratio= 0.1)

s = Serving(
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    model_key = "gemma",
    distribution_device = "cuda",
    distribution_type = "ddp",
    max_length = 2176,
    checkpoint_dir = os.path.join("checkpoints", current_time),
    result_dir = os.path.join("inference_outputs", current_time),
    torch_compile_config = {
            "torch_compile": True,
            "torch_compile_backend": "inductor",
            "torch_compile_mode": "default"
        }
)
s.inference(testdataset)