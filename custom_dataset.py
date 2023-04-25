import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
labels = {"good": 0, "bad": 1}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.labels = [labels[label] for label in data["label"]]
        self.text = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in data["text"]
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.text[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
