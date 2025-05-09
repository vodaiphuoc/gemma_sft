import os
os.environ["WANDB_DISABLED"] = "true"

from .commons import training

def main():
    training(
        model_id = "google-bert/bert-base-uncased", 
        train_path = os.path.dirname(__file__) + "/sample_data/traindata", 
        test_path = os.path.dirname(__file__) + "/sample_data/testdata",
        checkpoint_save_dir = os.path.join(os.path.dirname(__file__),"checkpoints"),
        num_train_epochs = 4,
        train_batch_size = 8,
        eval_batch_size = 8,
        learning_rate = 2e-4
    )

if __name__ == '__main__':
    main()