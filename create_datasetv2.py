from datasets import load_dataset, Dataset
import datasets
from typing import Set, List, Dict, Any, Literal
import itertools
from tqdm import tqdm
import json

datasets.disable_progress_bars()
SYSTEM_PROMPT:str = "You are a helpfull assistant always give emotional reponse in conservation"


def _get_unique_id(data: Dataset)->List[int]:
    unique_id = {}
    CacheDataset = data.select_columns(['conv_id', 'speaker_idx'])

    def _run(row):
        if unique_id.get(row['conv_id']) is None:
            unique_id[row['conv_id']] = set()
        else:
            if row['speaker_idx'] not in unique_id.get(row['conv_id']):
                unique_id.get(row['conv_id']).update(set([row['speaker_idx']]))

    CacheDataset = CacheDataset.map(lambda x: _run(x))

    return [
        _id
        for _id, _val 
        in unique_id.items() 
        if len(_val) == 2
    ]

def _process_history(conv_data: List[Dict[str, Any]])->Dict[str, List[Dict[str,str]]]:
    _history = [
        {
            "role": "user",
            "content": f"""
Conversation emotion: {conv_data[-1]['context']}
Situation: {conv_data[-1]['prompt'].replace('_comma_',',')}
{x['utterance'].replace('_comma_',',')}
""" 
        } if _ith%2 == 0 else \
        {
            "role": "model",
            "content": f"""
{x['utterance'].replace('_comma_',',')}
""" 
        }
        for _ith, x in enumerate(conv_data[: len(conv_data) -1])
    ]

    prompts = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }]

    prompts.extend(_history)

    return {
            "prompt": prompts,
            "completion": [{
                "role": "model",
                "content": f"{conv_data[-1]['utterance'].replace('_comma_',',')}"
            }]
        }

def _accum_handler(a, b):
    return a + [b]

def make_dataset(
        data: datasets.arrow_dataset.Dataset,
        dataset_type:str = Literal['train','valid','test'],
        version:str = "2.0"
    )->datasets.arrow_dataset.Dataset:

    init_unique_id_list = _get_unique_id(data= data)
    unique_conv_id = sorted(
        set(init_unique_id_list),
        key= lambda x: [int(ele.split(':')[-1]) for ele in x.split('_')]
    )
    print('number unique conv id: ', len(unique_conv_id))
    assert len(unique_conv_id) > 1, f"{dataset_type}"

    def _conv_gen(conv_id:str):
        conversation_data = data.filter(lambda x: x['conv_id'] == conv_id).sort("utterance_idx")
        _select_ids = list(range(2, len(conversation_data)+1, 2))
        accum_data = list(itertools.accumulate(conversation_data, _accum_handler, initial=[]))
        for _id in _select_ids:
            yield _process_history(accum_data[_id])

    def _dataset_generator(unique_conv_id: Set):
        # print(len(unique_conv_id), unique_conv_id[0])
        for _conv_id in tqdm(unique_conv_id, total = len(unique_conv_id)):
            _gen = _conv_gen(_conv_id)
            if _gen is not None:
                for ele in _gen:
                    yield ele
            else:
                continue
    
    print('building dataset')
    import time, os
    _start_time = time.time()
    new_data = Dataset.from_generator(
        _dataset_generator, 
        gen_kwargs = {'unique_conv_id': unique_conv_id},
        num_proc = 8
    )

    # for ele in new_data:
    #     print(json.dumps(ele, indent = 4))

    new_data.save_to_disk(os.path.join(os.path.dirname(__file__),"data",version,dataset_type))
    print('duration: ', time.time() - _start_time)

if __name__ == '__main__':
    VERSION = "3_0"
    # empathetic_dialogues
    data = load_dataset("facebook/empathetic_dialogues", trust_remote_code=True)
    traindata = data['train']
    valdata = data['validation']
    testdata = data['test']

    make_dataset(traindata, "train", VERSION)
    make_dataset(valdata, "valid", VERSION)
    make_dataset(testdata, "test", VERSION)