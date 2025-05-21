from .model import get_model_tokenizer
from .constants import DISTRIBUTION_DEVICE, DISTRIBUTION_TYPE
from peft import PeftModel
from datasets import Dataset
import torch
from torchmetrics.functional.text import bleu_score
from torchmetrics.functional.text.rouge import rouge_score
import os 

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
            distribution_type = distribution_type
        )
        merged_model = PeftModel.from_pretrained(base_model, checkpoint_dir).to(device)
        self.model = torch.compile(
            merged_model, 
            mode=torch_compile_config['torch_compile_mode'],
            backend=torch_compile_config['torch_compile_backend']
        )
        self.model.eval()

        self.result_dir = result_dir
        self.max_length = max_length
        print('done init')

    def _prepare_dataset(self, dataset: Dataset)->Dataset:
        dataset = dataset.select(list(range(12)))

        return dataset.map(
            lambda x: {
                "input_prompt":self.tokenizer.apply_chat_template(
                    x, 
                    tokenize = False,
                    add_generation_prompt= True
                )
            }
        )

    def inference(self, dataset: Dataset):
        dataset = self._prepare_dataset(dataset)
        print('done init test dataset')
        def _infer(row):
            print('batch prompt: ', len(row['input_prompt']))
            inputs = self.tokenizer(
                row['input_prompt'],
                add_special_tokens = False,
                padding = "max_length",
                truncation= True,
                max_length= self.max_length,
                padding_side = 'left',
                return_tensors = 'pt'
            ).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, 
                    **self._generation_config
                )

            return {
                "answer": self.tokenizer.batch_decode(outputs, skip_special_tokens = True)
            }

        dataset = dataset.map(
            lambda x: _infer(x), 
            batch_size = 6, 
            batched = True,
            desc="generating answers"
        )
        # save results
        dataset.to_json(os.path.join(self.result_dir,"prediction_results.json"))

        bleu_value = bleu_score(preds=dataset['answer'], target=dataset['completion'])
        rouge_value = rouge_score(preds=dataset['answer'], target=dataset['completion'])

        # get mean metrics
        report_metrics = {
                "bleu": bleu_value,
                "rouge1_fmeasure": rouge_value['rouge1_fmeasure'],
                "rouge2_fmeasure": rouge_value['rouge2_fmeasure'],
                "rougeL_fmeasure": rouge_value['rougeL_fmeasure']
            }
        
        print('testing result')
        print(report_metrics)