import os
from datasets import load_from_disk

def get_datasets(version:str = "2.0"):
    return (
        load_from_disk(os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"train")), 
        load_from_disk(os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"valid")),
        load_from_disk(os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"test")),
    )