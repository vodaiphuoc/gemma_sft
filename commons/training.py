import os
from .model import get_model_tokenizer
from .dataset import get_datasets
from .constants import (
    DISTRIBUTION_TYPE, 
    DISTRIBUTION_DEVICE,
)

def training_process(
        pre_init: tuple,
        model_key:str, 
        data_version:str,
        ratio: float,
        distribution_device: DISTRIBUTION_DEVICE,
        distribution_type: DISTRIBUTION_TYPE,
        checkpoint_save_dir:str,
        num_train_epochs:int = 4,
        train_batch_size:int = 8,
        eval_batch_size:int = 8,
        learning_rate: float = 2e-4,
        fsdp_config = None,
    ):

    # config diff between fsdp and ddp
    if distribution_type == "fsdp":
        os.environ["ACCELERATE_USE_FSDP"]= "true"
        torch_compile_config = {
            "torch_compile": False,
            "torch_compile_backend": None,
            "torch_compile_mode": None,
            "ddp_find_unused_parameters": False,
        }
        max_length = 1024
        dataloader_prefetch_factor = 3
        gradient_accumulation_steps = 6
    else:
        torch_compile_config = {
            "torch_compile": True,
            "torch_compile_backend": "inductor",
            "torch_compile_mode": "default",
            "ddp_find_unused_parameters": True,
        }
        max_length = 640 if model_key == "gemma" else 512
        dataloader_prefetch_factor = 2
        gradient_accumulation_steps = 3

    import numpy as np
    from torchmetrics.functional.text import bleu_score
    from torchmetrics.functional.text.rouge import rouge_score
    from trl import SFTConfig, SFTTrainer

    if pre_init is None:
        model, tokenizer, lora_config = get_model_tokenizer(
            model_key = model_key,
            distribution_device = distribution_device,
            distribution_type = distribution_type
        )
    else:
        model, tokenizer, lora_config = pre_init

    converted_traindata, converted_validdata, converted_testdata = get_datasets(data_version, ratio)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
    
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        
        bleu_value = bleu_score(preds=decoded_preds, target=decoded_labels)
        rouge_value = rouge_score(preds=decoded_preds, target=decoded_labels)
        return {
            "bleu": bleu_value,
            "rouge1_fmeasure": rouge_value['rouge1_fmeasure'],
            "rouge2_fmeasure": rouge_value['rouge2_fmeasure'],
            "rougeL_fmeasure": rouge_value['rougeL_fmeasure']
        }
    
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = converted_traindata,
        eval_dataset = converted_validdata,
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        args = SFTConfig(
            do_train = True,
            do_eval = True,
            eval_strategy = 'epoch',
            save_strategy = 'no',
            num_train_epochs = num_train_epochs,
            per_device_train_batch_size = train_batch_size,
            per_device_eval_batch_size = eval_batch_size,
            dataset_num_proc = 4,
            dataloader_pin_memory = True,
            dataloader_drop_last=True,
            dataloader_num_workers = 2,
            dataloader_prefetch_factor = dataloader_prefetch_factor,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps=2,
            completion_only_loss = True,
            learning_rate=learning_rate,
            fp16=True,
            fp16_full_eval = True,
            max_length = max_length,
            packing = True,
            eval_packing = False,
            jit_mode_eval = False,
            max_seq_length = None,
            lr_scheduler_type = 'cosine_with_restarts',
            lr_scheduler_kwargs = {"num_cycles": 10},
            optim = 'adamw_torch_fused',
            label_names=["labels"],
            logging_strategy = 'epoch',
            report_to = "none",
            output_dir = checkpoint_save_dir,
            fsdp = fsdp_config['fsdp_sharding_strategy'].lower() if fsdp_config is not None else '',
            fsdp_config = fsdp_config,
            **torch_compile_config
        ),
        peft_config=lora_config, # lora config
    )

    print('start training')
    trainer.train()
    print('run evaluate')
    output_metrics = trainer.evaluate()
    print('output metrics: ', output_metrics)
    print('done training, saving model')
    trainer.save_model(checkpoint_save_dir)