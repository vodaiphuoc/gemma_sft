import os

from .model import get_model_tokenizer
from .dataset import get_datasets

def training_process(
        model_id:str, 
        data_version:str,
        checkpoint_save_dir:str,
        use_lora: bool,
        num_train_epochs:int = 4,
        train_batch_size:int = 8,
        eval_batch_size:int = 8,
        learning_rate: float = 2e-4
    ):
    os.environ["ACCELERATE_USE_FSDP"]= "true"
    
    model, tokenizer = get_model_tokenizer(model_id = model_id)

    converted_traindata, converted_validdata, converted_testdata = get_datasets(data_version)

    import numpy as np
    from torchmetrics.functional.text import bleu_score
    from torchmetrics.functional.text.rouge import rouge_score

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
            "rouge": rouge_value
        }
    
    from trl import SFTConfig, SFTTrainer

    if use_lora:
        from peft import LoraConfig
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha = 8,
            lora_dropout = 0.05,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
    else:
        lora_config = None
    
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
            # num_train_epochs = num_train_epochs,
            max_steps = 20,
            per_device_train_batch_size = train_batch_size,
            per_device_eval_batch_size = eval_batch_size,
            dataset_num_proc = 2,
            gradient_accumulation_steps = 4,
            warmup_steps=2,
            completion_only_loss = True,
            learning_rate=learning_rate,
            bf16=True,
            bf16_full_eval = True,
            max_length = 640,
            packing = True,
            max_seq_length = 128,
            optim = 'adamw_torch_fused',
            label_names=["labels"],
            logging_strategy = 'no',
            report_to = None
        ),
        peft_config=lora_config, # lora config
    )
    print('start training')
    trainer.train()
    print('done training, saving model')
    trainer.save_model(checkpoint_save_dir)
    # print('run evaluate')
    # output_metrics = trainer.evaluate(converted_testdata)
    # print('output metrics: ', output_metrics)
