import os
from datasets import load_from_disk

def get_datasets(version:str = "2.0"):
    train_folder = os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"train")
    print(train_folder)

    print(os.path.isdir(train_folder))

    traindata = load_from_disk(), 
    valdata = load_from_disk(os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"valid")),
    testdata = load_from_disk(os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"test")),

    print(f"""
- load data version {version}
- train size: {len(traindata)}
- valid size: {len(valdata)}
- test size: {len(testdata)}
""")
    return traindata, valdata, testdata