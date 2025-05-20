from .model import get_model_tokenizer
from .constants import DISTRIBUTION_DEVICE, DISTRIBUTION_TYPE
from peft import PeftModel
from datasets import Dataset
from trl import apply_chat_template
import torch
from torchmetrics.functional.text import bleu_score
from torchmetrics.functional.text.rouge import rouge_score

class Serving(object):
    _generation_config = {
        "max_new_tokens": 60,
        "do_sample": True,
        "num_beams": 2,
        "temperature": 0.1,
        "length_penalty": -0.2
    }

    def __init__(
            self,
            device: torch.device, 
            model_key:str,
            distribution_device: DISTRIBUTION_DEVICE,
            distribution_type: DISTRIBUTION_TYPE,
            max_length:int,
            checkpoint_dir:str,
            result_dir: str
            ):
        base_model, self.tokenizer, _ = get_model_tokenizer(
            model_key= model_key, 
            distribution_device= distribution_device, 
            distribution_type = distribution_type
        )
        self.model = PeftModel.from_pretrained(base_model, checkpoint_dir).to(device)
        self.result_dir = result_dir
        self.max_length = max_length

    def _prepare_dataset(self, dataset: Dataset)->Dataset:
        return dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer}
        )

    def inference(self, dataset: Dataset):
        dataset = self._prepare_dataset(dataset)

        def _infer(row):
            inputs = self.tokenizer(
                row['prompt'],
                add_special_tokens = False,
                padding = "max_length",
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
                "answer": self.tokenizer.batch_decode(outputs)
            }

        dataset = dataset.map(lambda x: _infer(x), batch_size = 8, batched = True)
        # save results
        dataset.to_json(self.result_dir)

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
            