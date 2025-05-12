import os

from commons.model import get_model_tokenizer
from commons.training import training_process
from commons.constants import DISTRIBUTION_TYPES, MODEL_KEY2IDS
from commons.utils import get_fsdp_config_from_yaml
import argparse

def main(
        distribution_type: DISTRIBUTION_TYPES, 
        model_key:str, 
        fsdp_config_path:str,
        pre_init: tuple = None
    ):
    assert MODEL_KEY2IDS.get(model_key) is not None
    fsdp_config = get_fsdp_config_from_yaml(fsdp_config_path)
    if distribution_type == "tpu":
        assert fsdp_config is not None

    training_process(
        pre_init = pre_init,
        model_key = model_key, 
        data_version = "2_0",
        ratio = 0.01,
        distribution_type = distribution_type,
        checkpoint_save_dir = os.path.join(os.path.dirname(__file__),"checkpoints"),
        num_train_epochs = 3,
        train_batch_size = 12,
        eval_batch_size = 12,
        learning_rate = 1e-5,
        fsdp_config = fsdp_config
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution_type', type=str)
    parser.add_argument('--model_key', type=str)
    parser.add_argument('--fsdp_config_path', type=str, default= "")
    args = parser.parse_args()

    if args.distribution_type == "tpu":
        model, tokenizer, lora_config = get_model_tokenizer(
            model_key= args.model_key,
            distribution_type = args.distribution_type
        )
        
        os.environ['PJRT_DEVICE'] = 'TPU'
        from accelerate import notebook_launcher
        notebook_launcher(
            main,
            (
                args.distribution_type,
                args.model_key, 
                args.fsdp_config_path, 
                (model, tokenizer, lora_config)
            ), 
            num_processes = 8
        )
    else:
        main(
            distribution_type = args.distribution_type,
            model_key= args.model_key,
            fsdp_config_path = args.fsdp_config_path
        )
