import os
from commons.training import training_process
from commons.constants import DISTRIBUTION_TYPES, MODEL_KEY2IDS
import argparse

def main(distribution_type: DISTRIBUTION_TYPES, model_key:str):
    assert MODEL_KEY2IDS.get(model_key) is not None
    training_process(
        model_key = model_key, 
        data_version = "2_0",
        ratio = 0.01,
        distribution_type = distribution_type,
        checkpoint_save_dir = os.path.join(os.path.dirname(__file__),"checkpoints"),
        num_train_epochs = 3,
        train_batch_size = 12,
        eval_batch_size = 12,
        learning_rate = 1e-5
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distribution_type', type=str)
    parser.add_argument('--model_key', type=str)
    args = parser.parse_args()

    main(
        distribution_type = args.distribution_type,
        model_key= args.model_key
    )
