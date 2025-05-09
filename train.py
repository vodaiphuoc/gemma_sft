
import os
os.environ["WANDB_DISABLED"] = "true"


def get_model_tokenizer():
    # from trl import (
    #     get_kbit_device_map,
    #     get_peft_config,
    #     get_quantization_config,
    # )
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig
    )
    
    model_id = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # import torch
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    from accelerate import PartialState
    device_string = PartialState().process_index
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        # quantization_config=bnb_config,
        attn_implementation='eager',
        device_map={'':device_string}
    )
    return model, tokenizer

def get_datasets(
        train_path: str = "./data/traindata", 
        test_path:str = "./data/testdata"
    ):
    from datasets import load_from_disk
    return load_from_disk(train_path), load_from_disk(test_path)

def main():
    os.environ["ACCELERATE_USE_FSDP"]= "true"
    
    model, tokenizer = get_model_tokenizer()

    converted_traindata, converted_testdata = get_datasets(
        train_path = os.path.dirname(__file__) + "/sample_data/traindata", 
        test_path = os.path.dirname(__file__) + "/sample_data/testdata"
    )

    import numpy as np
    from torchmetrics.functional.text import bleu_score

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
        
        result = bleu_score(preds=decoded_preds, target=decoded_labels)
        return {"bleu": result}
    
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = converted_traindata,
        eval_dataset = converted_testdata,
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        args = SFTConfig(
            do_train = True,
            do_eval = False,
            num_train_epochs = 4,
            per_device_train_batch_size = 4,
            per_device_eval_batch_size = 4,
            gradient_accumulation_steps= 3,
            warmup_steps=2,
            completion_only_loss = True,
            learning_rate=2e-4,
            bf16=True,
            bf16_full_eval = True,
            max_length = 128,
            max_seq_length = 128,
            logging_strategy = 'epoch',
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        peft_config=lora_config, # lora config
    )
        
    trainer.train()
    trainer.save_model(os.path.join(os.path.dirname(__file__),"checkpoints"))
    trainer.evaluate()

if __name__ == '__main__':
    main()