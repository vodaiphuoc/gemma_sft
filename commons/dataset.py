def get_datasets(
        train_path:str = "./data/traindata", 
        test_path:str = "./data/testdata"
    ):
    from datasets import load_from_disk
    return load_from_disk(train_path), load_from_disk(test_path)
