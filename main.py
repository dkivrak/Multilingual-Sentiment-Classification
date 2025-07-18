from datasets import load_dataset
from model import Model


datasets = load_dataset("csv", data_files={
    "train": "data/train.csv",
    "validation": "data/valid.csv"
})


model = Model()


training_args = {

    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
}


model.train(datasets, training_args)


model.save("saved_objects")
