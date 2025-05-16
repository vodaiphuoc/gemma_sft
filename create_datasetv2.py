from datasets import load_dataset, Dataset
import datasets
from typing import Set, List, Dict, Any, Literal
import itertools
from tqdm import tqdm

datasets.disable_progress_bars()
SYSTEM_PROMPT:str = "You are a helpfull assistant always give emotional reponse in conservation"

def _process_history(conv_data: List[Dict[str, Any]])->Dict[str, List[Dict[str,str]]]:
    _history = [
        f"SpeakerID_{x['speaker_idx']}: {x['utterance'].replace('_comma_',',')}\n"
        for x in conv_data[: len(conv_data) -1]
    ]

    user_content = f"""
Conversation context: {conv_data[-1]['context']}
Context: {conv_data[-1]['prompt'].replace('_comma_',',')}
Conversation history:
{''.join(_history)}
Base on context and history of above conversation, please give emotional and appropriate reponse.
""" 
    return {
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
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
    unique_conv_id = sorted(set(data['conv_id']),key= lambda x: [ele[-1] for ele in x.split('_')])
    print('number unique conv id: ', len(unique_conv_id))

    def _conv_gen(conv_id:str):
        conversation_data = data.filter(lambda x: x['conv_id'] == conv_id).sort("utterance_idx")
        for ele in list(itertools.accumulate(conversation_data, _accum_handler, initial=[]))[2:]:
            yield _process_history(ele)

    def _dataset_generator(unique_conv_id: Set):
        print(len(unique_conv_id), unique_conv_id[0])
        for _conv_id in tqdm(unique_conv_id, total = len(unique_conv_id)):
            for ele in _conv_gen(_conv_id):
                yield ele

    import time, os
    _start_time = time.time()
    new_data = Dataset.from_generator(
        _dataset_generator, 
        gen_kwargs = {'unique_conv_id': unique_conv_id},
        num_proc = 4
    )

    new_data.save_to_disk(os.path.join(os.path.dirname(__file__),"data",version,dataset_type))
    print('duration: ', time.time() - _start_time)

if __name__ == '__main__':
    VERSION = "2_1"
    # empathetic_dialogues
    data = load_dataset("facebook/empathetic_dialogues", trust_remote_code=True)
    traindata = data['train']
    valdata = data['validation']
    testdata = data['test']

    make_dataset(traindata, "train", VERSION)
    make_dataset(valdata, "valid", VERSION)
    make_dataset(testdata, "test", VERSION)