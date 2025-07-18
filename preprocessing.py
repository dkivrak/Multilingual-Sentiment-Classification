from datasets import DatasetDict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer

class Preprocessor:
    def __init__(self, tokenizer, collator, **kwargs):
        self.tokenizer = tokenizer
        self.collator  = collator
        self.label_encoder = LabelEncoder(labels = {"negative": 0, "positive": 1})
        self.__dict__.update(kwargs)

    def prepare_data(self, dataset: DatasetDict) -> DatasetDict:
        raise NotImplementedError

class LabelEncoder:
    def __init__(self, labels, **kwargs):
        self.id2label = {v: k for k, v in labels.items()}
        self.label2id = {k: v for k, v in labels.items()}
        self.__dict__.update(kwargs)


class Tokenizer:
    def __init__(self, model_checkpoint: str = "distilbert-base-multilingual-cased", **kwargs):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def train(self, texts: List[str]):
        pass

    def tokenize(self, text: str) -> Dict:
        return self.tokenizer(text, truncation=True, padding="max_length")

    def push_to_hub(self, path: str):
        self.tokenizer.push_to_hub(path)

    def from_pretrained(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        return self


class Embedder:
    def __init__(self, **kwargs):
        pass

    def embed(self, inputs):
        raise NotImplementedError("Embedder functionality not implemented.")

    def push_to_hub(self, path: str):
        pass

    def from_pretrained(self, path: str):
        return self
