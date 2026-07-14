import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class STSDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # instruction = "Given a text, Retrieve semantically similar text: "
        instruction=""
        return {
            "sentence1": instruction + self.dataset.iloc[idx]['sentence1'],
            "sentence2": instruction + self.dataset.iloc[idx]['sentence2'],
            "label": torch.tensor(self.dataset.iloc[idx]['score'], dtype=torch.float),
        }

    def collate_fn(self, batch, tokenizer, max_len=256):
        s1_list = [item["sentence1"] for item in batch]
        s2_list = [item["sentence2"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])

        enc1 = tokenizer(
            s1_list,
            truncation=True,
            padding=True,       # chỉ pad theo câu dài nhất trong batch
            max_length=max_len,
            return_tensors="pt"
        )
        enc2 = tokenizer(
            s2_list,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )

        return {
            "input_ids1": enc1["input_ids"],
            "attention_mask1": enc1["attention_mask"],
            "input_ids2": enc2["input_ids"],
            "attention_mask2": enc2["attention_mask"],
            "labels": labels,
        }


class ClasssifyDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "text": self.dataset.iloc[idx]['text'],
            "label": torch.tensor(self.dataset.iloc[idx]['label'], dtype=torch.long),
        }

    def collate_fn(self, batch, tokenizer, max_len=256):
        s1_list = [item["text"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])

        enc1 = tokenizer(
            s1_list,
            truncation=True,
            padding=True,       # chỉ pad theo câu dài nhất trong batch
            max_length=max_len,
            return_tensors="pt"
        )

        return {
            "input_ids1": enc1["input_ids"],
            "attention_mask1": enc1["attention_mask"],
            "labels": labels,
        }


class PairDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # instruction = "Given a text, Retrieve semantically similar text: "
        instruction=""
        return {
            "sentence1": instruction + self.dataset.iloc[idx]['sentence1'],
            "sentence2": instruction + self.dataset.iloc[idx]['sentence2'],
            "label": torch.tensor(self.dataset.iloc[idx]['label'], dtype=torch.float),
        }
    
    def collate_fn(self, batch, tokenizer, max_len=256):
        s1_list = [item["sentence1"] for item in batch]
        s2_list = [item["sentence2"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])

        enc1 = tokenizer(
            s1_list,
            truncation=True,
            padding=True,       # chỉ pad theo câu dài nhất trong batch
            max_length=max_len,
            return_tensors="pt"
        )
        enc2 = tokenizer(
            s2_list,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids1": enc1["input_ids"],
            "attention_mask1": enc1["attention_mask"],
            "input_ids2": enc2["input_ids"],
            "attention_mask2": enc2["attention_mask"],
            "labels": labels,
        }
    