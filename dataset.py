from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import json

from torch.utils.data import DataLoader, Dataset

class BiEncoderDataset(Dataset):
    def __init__(self, file_path, teacher_embedding_path):
        self.dataset = pd.read_csv(file_path)
        self.teacher_embedding = torch.load(teacher_embedding_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        query = self.dataset.iloc[index]['text']
        teacher_embedding = self.teacher_embedding[index]
        return query, teacher_embedding


@dataclass
class BiDataCollator:
    student_tokenizer: PreTrainedTokenizer = None
    max_len: int = 64
    return_tensors: str = 'pt'
    padding: bool = True


    def __call__(self, batch):
        queries = []
        teacher_embeddings = []
        for q, t_embed in batch:
            queries.append(q)
            teacher_embeddings.append(t_embed)

        teacher_embeddings = torch.stack(teacher_embeddings)

        student_inputs = self.student_tokenizer(
            queries,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
        )

        return (student_inputs, teacher_embeddings)
    

def build_dataloader(train_data_path, teacher_embedding_path, student_tokenizer, max_len, batch_size):
    dataset = BiEncoderDataset(train_data_path, teacher_embedding_path)
    data_collator = BiDataCollator(student_tokenizer=student_tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    return dataset, dataloader