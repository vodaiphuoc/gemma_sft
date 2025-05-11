import os
from commons.training import training_process

def main():
    training_process(
        model_key = "gemma_unsloth", 
        data_version = "2_0",
        ratio = 0.01,
        checkpoint_save_dir = os.path.join(os.path.dirname(__file__),"checkpoints"),
        num_train_epochs = 3,
        train_batch_size = 12,
        eval_batch_size = 12,
        learning_rate = 1e-5
    )

if __name__ == '__main__':
    main()
