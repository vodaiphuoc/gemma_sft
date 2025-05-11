import os
from datasets import load_from_disk

def get_datasets(version:str = "2_0", ratio: float = None):
    traindata = load_from_disk(os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"train"))
    valdata = load_from_disk(os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"valid"))
    testdata = load_from_disk(os.path.join(os.path.dirname(__file__).replace("commons","data"),version,"test"))

    if ratio is not None:
        traindata = traindata.select(list(range(int(len(traindata)*ratio ))))
        valdata = valdata.select(list(range(int(len(valdata)*ratio ))))
        testdata = testdata.select(list(range(int(len(testdata)*ratio ))))

    print(f"""
- load data version {version}
- ratio: {ratio}
- train size: {len(traindata)}
- valid size: {len(valdata)}
- test size: {len(testdata)}
""")
    return traindata, valdata, testdata