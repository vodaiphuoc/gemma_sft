# a mock trainer from trl for debugging
import datasets
from typing import Union, Optional, Any, List
import dataclasses
from trl import SFTTrainer, SFTConfig
from transformers.utils import is_datasets_available
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers.data.data_collator import DataCollatorMixin, PreTrainedTokenizerBase
from .utils import pad

@dataclasses.dataclass
class MockDataCollatorForLanguageModeling(DataCollatorMixin):
    
    pad_token_id: int
    completion_only_loss: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]
        labels = [torch.tensor(example["input_ids"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]

        # Pad
        output = {}
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        output["labels"] = pad(
            labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = pad(
                completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion

        return output



class MockSFTTrainer(SFTTrainer):
    r"""
    A mock class wrapper of `SFTTrainer` to resolve packing and completion loss only
    """
    _COLUMN_NAMES = ["input_ids", "attention_mask", "completion_mask"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print('check mix precision of accelerator: ',self.accelerator.mixed_precision)
        print('self.model type: ', type(self.model))
        print('num trainable params: ',self.model.print_trainable_parameters())

    def _find_chunk_ids(
            self, 
            dataset: datasets.Dataset, 
            max_length:int = 1024,
            sum_val = 0
        )->List[List[int]]:
        dataset = dataset.map(
            lambda x: {
                "seq_length": len(x['input_ids'])
            }
        )

        chunk_ids = []
        buffer = []

        for ith, ele in enumerate(dataset['seq_length']):
            if sum_val + ele > max_length:
                chunk_ids.append(buffer)
                buffer = [ith]
                sum_val = ele
            else:
                sum_val += ele
                buffer.append(ith)
        chunk_ids.append(buffer)
        return chunk_ids

    def _prepare_dataset(self, *args, **kwargs)->datasets.Dataset:
        dataset = super()._prepare_dataset(*args, **kwargs)

        # get sft config
        config = None
        dataset_name = None
        for _arg in args:
            if isinstance(_arg, SFTConfig):
                config = _arg
            elif isinstance(_arg,str) and _arg in ['train','eval']:
                dataset_name = _arg
            else:
                continue

        for _, _kwarg_value in kwargs.items():
            if isinstance(_kwarg_value, SFTConfig):
                config = _kwarg_value
            elif isinstance(_arg,str) and _arg in ['train','eval']:
                dataset_name = _kwarg_value
            else:
                continue

        if dataset_name == 'eval':
            print(f'skip packing for {dataset_name} dataset')
            return dataset
        else:
            assert dataset_name == "train"
            print(f'running packing for {dataset_name} dataset')
            print('check output dataset col names: ',dataset.column_names)
            dataset = dataset.select_columns(self._COLUMN_NAMES)

            chunk_ids = self._find_chunk_ids(
                dataset = dataset, 
                max_length = config.max_length
            )
            print('get number of chunk:', len(chunk_ids))
            def _dataset_gen(chunk_ids: List[List[int]]):
                for chunk in chunk_ids:
                    chunk_dataset = dataset.select(chunk)
                    yield {
                        col: [ele for each_ids in chunk_dataset[col] for ele in each_ids] \
                        if col != "completion_mask" else \
                        [1 for each_ids in chunk_dataset[col] for ele in each_ids]
                        for col in self._COLUMN_NAMES
                    }

            packed_dataset = datasets.Dataset.from_generator(
                _dataset_gen, 
                gen_kwargs = {"chunk_ids": chunk_ids}
            )
            print('length dataset after packing: ', len(packed_dataset))
            return packed_dataset.select_columns(self._COLUMN_NAMES)

class MockSFTTrainerV2(SFTTrainer):
    def __init__(
            self,
            processing_class: PreTrainedTokenizerBase,
            *args,
            **kwargs
        ):
        super().__init__(processing_class = processing_class, *args, **kwargs)
        pad_token = processing_class.pad_token
        self._tokenizer_pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
        self.model.print_trainable_parameters()

    def get_eval_dataloader(
            self, 
            eval_dataset: datasets.Dataset = None,
        ) -> DataLoader:
        """
        Overide default method
        """
        data_collator = MockDataCollatorForLanguageModeling(self._tokenizer_pad_token_id)
        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)



class MockSaveTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('self.model type: ', type(self.model))
        print('num trainable params: ',self.model.print_trainable_parameters())

    def save_model(self, output_dir: str):
        r"""Overide internal save_model method"""
        from torchao.quantization.qat import (
            Int8DynActInt4WeightQATQuantizer
        )
        import os
        quantizer = Int8DynActInt4WeightQATQuantizer(groupsize= 32)
        
        print('current model: ', self.model)
        print('current model type: ', type(self.model))
        
        merged_model = self.model.merge_and_unload()
        
        quanted_model = quantizer.convert(merged_model)
        print('quanted_model: ', quanted_model)
        print('quanted_model type: ', type(quanted_model))
        
        os.mkdir(os.path.join(output_dir, "state_dict"))
        os.mkdir(os.path.join(output_dir, "full_model"))
        torch.save(quanted_model.state_dict(), os.path.join(output_dir, "state_dict"))
        torch.save(quanted_model, os.path.join(output_dir, "full_model"))