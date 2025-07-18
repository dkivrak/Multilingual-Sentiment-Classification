import pickle

from datasets import DatasetDict
from typing import List, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score
import os

from preprocessing import Embedder


class Model:
    def __init__(self, pretrained_path: str = None, model_checkpoint: str = "distilbert-base-multilingual-cased", **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        if pretrained_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path).to(self.device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(self.device)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    def train(self, datasets: List[DatasetDict], training_args: Dict):
        tokenized_datasets = datasets.map(self.tokenize_function, batched=True)

        args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            **training_args
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {"f1": f1_score(labels, predictions, average="macro")}

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        self.trainer = trainer

    def predict(self, x: List[List[float]]) -> List[int]:
        inputs = self.tokenizer(x, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        return preds.cpu().numpy().tolist()

    def compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        f1 = f1_score(y_true, y_pred, average="macro")
        return {"f1": f1}

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self, f)


        with open(os.path.join(path, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.tokenizer, f)

        embedder = Embedder()
        with open(os.path.join(path, "embedder.pkl"), "wb") as f:
            pickle.dump(embedder, f)

    def load(self, path):
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        return self
