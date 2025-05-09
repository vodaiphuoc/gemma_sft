import os
os.environ["WANDB_DISABLED"] = "true"

from commons.training import training_process

def main():
    training_process(
        model_id = "google/gemma-3-1b-it", 
        train_path = os.path.dirname(__file__) + "/sample_data/traindata", 
        test_path = os.path.dirname(__file__) + "/sample_data/testdata",
        checkpoint_save_dir = os.path.join(os.path.dirname(__file__),"checkpoints"),
        use_lora = True,
        num_train_epochs = 4,
        train_batch_size = 8,
        eval_batch_size = 8,
        learning_rate = 2e-4
    )

if __name__ == '__main__':
    main()
