from datasets import load_dataset
import datasets
# empathetic_dialogues
data = load_dataset("facebook/empathetic_dialogues", trust_remote_code=True)
traindata = data['train'].select(list(range(15000)))
testdata = data['test'].select(list(range(5000)))


def make_dataset(
        data: datasets.arrow_dataset.Dataset, 
        system_prompt:str = "You are a helpfull assistant always give emotional reponse in conservation"
    )->datasets.arrow_dataset.Dataset:
    r"""
    Example of conversation data in prompt-completion format:
    prompt_completion_example = {
        "prompt": [{"role": "user", "content": "What color is the sky?"}],
        "completion": [{"role": "assistant", "content": "It is blue."}]
    }
    """
    
    def _handler(row: dict):
        history = None
        if row['utterance_idx'] > 1:
            history_data = data.filter(
                lambda x: x['conv_id'] == row['conv_id'] and \
                        x['utterance_idx'] < row['utterance_idx']
            )
            if len(history_data) == 0:
                history = []
            else:
                history = history_data.map(
                    lambda x: {
                        'content': f"SpeakerID_{x['speaker_idx']}: {x['utterance']}\n" 
                    }
                )['content']
            
        else:
            history = []

        user_content = f"""
Conversation context: {row['context']}
Context: {row['prompt'].replace('_comma_',',')}
Conversation history:
{''.join(history)}
"""
        if len(history) > 0:
            messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                    
            ]
        else:
            messages = None
        return {
            "prompt": messages,
            "completion": [{
                "role": "model",
                "content": f"{row['utterance']}"
            }]
        }

    data = data.map(
        lambda x: _handler(x),
        batch_size = 1000,
        batched = False
    )
    data = data.filter(lambda x: x['prompt'] is not None)
    return data.select_columns(['prompt', 'completion'])

if __name__ == '__main__':
    import time
    import os

    _start_time = time.time()
    datasets.disable_progress_bars()
    converted_testdata = make_dataset(testdata)
    converted_traindata = make_dataset(traindata)
    print('total pre-process data time: ', time.time() - _start_time)

    
    converted_testdata.save_to_disk(os.path.join(os.path.dirname(__file__),"data","testdata"))
    converted_traindata.save_to_disk(os.path.join(os.path.dirname(__file__),"data","traindata"))