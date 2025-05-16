# a mock trainer from trl for debugging
from datasets import Dataset
from typing import Union, Optional, Any, List

from trl import SFTTrainer, SFTConfig

class MockSFTTrainer(SFTTrainer):
    r"""
    A mock class wrapper of `SFTTrainer` to resolve packing and completion loss only
    """
    _COLUMN_NAMES = ["input_ids", "attention_mask", "completion_mask"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def _find_chunk_ids(
            self, 
            dataset: Dataset, 
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

    def _prepare_dataset(self, *args, **kwargs)->Dataset:
        dataset = super()._prepare_dataset(*args, **kwargs)

        # get sft config
        config = None
        dataset_name = None
        for _arg in args:
            if isinstance(_arg, SFTConfig):
                config = _arg
            if isinstance(_arg,str) and _arg in ['train','val']:
                dataset_name = _arg

        for _, _kwarg_value in kwargs.items():
            if isinstance(_kwarg_value, SFTConfig):
                config = _kwarg_value
            if isinstance(_arg,str) and _arg in ['train','val']:
                dataset_name = _arg
        
        if dataset_name == 'val':
            print(f'skip packing for {dataset_name} dataset')
            return dataset
        else:
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
                        col: [ele for each_ids in chunk_dataset[col] for ele in each_ids]
                        for col in self._COLUMN_NAMES
                    }

            packed_dataset = Dataset.from_generator(
                _dataset_gen, 
                gen_kwargs = {"chunk_ids": chunk_ids}
            )
            print('lenght packed dataset: ', packed_dataset)
            return packed_dataset.select_columns(self._COLUMN_NAMES)