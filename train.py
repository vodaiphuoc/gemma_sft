import os
import argparse
from commons.constants import DISTRIBUTION_TYPE, DISTRIBUTION_DEVICE, MODEL_KEY2IDS
from commons.model import get_model_tokenizer

def main(
        distribution_device: DISTRIBUTION_DEVICE, 
        distribution_type: DISTRIBUTION_TYPE, 
        model_key:str, 
        fsdp_config_path:str,
        train_batch_size: int, 
        eval_batch_size: int,
        pre_init: tuple = None
    ):
    
    from commons.training import training_process
    from commons.utils import get_fsdp_config_from_yaml

    assert MODEL_KEY2IDS.get(model_key) is not None
    fsdp_config = get_fsdp_config_from_yaml(fsdp_config_path)
    if distribution_type == "tpu":
        assert fsdp_config is not None
    
    training_process(
        pre_init = pre_init,
        model_key = model_key,
        data_version = "2_1",
        ratio = 0.01,
        distribution_device = distribution_device,
        distribution_type = distribution_type,
        logging_dir = os.path.join(os.path.dirname(__file__),"logs"),
        checkpoint_save_dir = os.path.join(os.path.dirname(__file__),"checkpoints"),
        num_train_epochs = 2,
        train_batch_size = train_batch_size,
        eval_batch_size = eval_batch_size,
        learning_rate = 1e-4,
        fsdp_config = fsdp_config
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution_device', type=str)
    parser.add_argument('--distribution_type', type=str)
    parser.add_argument('--model_key', type=str)
    parser.add_argument('--train_batch_size', type=int, default= 12)
    parser.add_argument('--eval_batch_size', type=int, default= 12)
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
            distribution_device = args.distribution_device,
            distribution_type = args.distribution_type,
            model_key= args.model_key,
            fsdp_config_path = args.fsdp_config_path,
            train_batch_size = args.train_batch_size, 
            eval_batch_size = args.eval_batch_size,
        )
