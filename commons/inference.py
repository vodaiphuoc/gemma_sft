from .model import get_model_tokenizer
from .constants import DISTRIBUTION_DEVICE, DISTRIBUTION_TYPE
from peft import PeftModel
from trl import apply_chat_template
from datasets import Dataset
import torch
from torchmetrics.functional.text import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
import os 
import json

class Serving(object):
    _generation_config = {
        "max_new_tokens": 64,
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95
    }

    def __init__(
            self,
            device: torch.device, 
            model_key:str,
            distribution_device: DISTRIBUTION_DEVICE,
            distribution_type: DISTRIBUTION_TYPE,
            max_length:int,
            checkpoint_dir:str,
            result_dir: str,
            torch_compile_config: dict
            ):
        print("Serving init on device: ", device)
        base_model, self.tokenizer, _ = get_model_tokenizer(
            model_key= model_key, 
            distribution_device= distribution_device, 
            distribution_type = distribution_type,
            checkpoint_dir= checkpoint_dir
        )
        base_model = base_model.to(torch.float16)
        # merged_model = PeftModel.from_pretrained(base_model, checkpoint_dir).to(device)

        # print('model after merge: ', merged_model)

        self.model = torch.compile(
            base_model, 
            mode=torch_compile_config['torch_compile_mode'],
            backend=torch_compile_config['torch_compile_backend']
        )
        self.model.eval()

        self.result_dir = result_dir
        self.max_length = max_length
        print('done init')

    def _tokenize(self, row):
        outputs = apply_chat_template(row, self.tokenizer)
        return {
            "input_prompt": outputs['prompt'],
            "desired_completion": outputs['completion'],
        }

    def _prepare_dataset(self, dataset: Dataset)->Dataset:
        dataset = dataset.select(list(range(12)))
        return dataset.map(
            lambda x: self._tokenize(x),
            keep_in_memory = True,
        )

    def inference(self, dataset: Dataset):
        dataset = self._prepare_dataset(dataset)
        print('done init test dataset')

        def _infer(row):
            inputs = self.tokenizer(
                row['input_prompt'],
                add_special_tokens = False,
                padding = "max_length",
                truncation= True,
                max_length= self.max_length,
                padding_side = 'left',
                return_tensors = 'pt'
            ).to(self.model.device)
            print('input ids length: ', inputs['input_ids'].shape)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, 
                    **self._generation_config
                )
            
            return {
                "model_response": self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens = True)
            }

        dataset = dataset.map(
            lambda x: _infer(x), 
            batch_size = 6, 
            batched = True,
            desc="generating answers"
        )
        # save results
        save_dict = [
            {
                "original_prompt": ele['prompt'],
                "input_prompt": ele['input_prompt'],
                "desired_completion": ele['desired_completion'],
                "model_response": ele['model_response'],
            }   
            for ele in dataset
        ]

        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        with open(os.path.join(self.result_dir,"prediction_results.json"),'w') as fp:
            json.dump(save_dict, fp, indent= 4)

        bleu_value = bleu_score(preds=dataset['model_response'], target=dataset['desired_completion'])
        rouge_value = rouge_score(preds=dataset['model_response'], target=dataset['desired_completion'])

        # get mean metrics
        report_metrics = {
                "bleu": bleu_value,
                "rouge1_fmeasure": rouge_value['rouge1_fmeasure'],
                "rouge2_fmeasure": rouge_value['rouge2_fmeasure'],
                "rougeL_fmeasure": rouge_value['rougeL_fmeasure']
            }
        
        print('testing result')
        print(report_metrics)