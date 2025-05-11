import os
os.environ["WANDB_DISABLED"] = "true"

from commons.training import training_process

def main():
    training_process(
        model_id = "google/gemma-3-1b-it", 
        data_version = "2_0",
        ratio = 0.05,
        checkpoint_save_dir = os.path.join(os.path.dirname(__file__),"checkpoints"),
        use_lora = True,
        num_train_epochs = 3,
        train_batch_size = 10,
        eval_batch_size = 8,
        learning_rate = 1e-5
    )

if __name__ == '__main__':
    main()
