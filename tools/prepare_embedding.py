import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare vector embeddings"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="data train",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Tên hoặc đường dẫn model",
    )

    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], idx  # trả về text và index để map lại

def collate_fn(batch, tokenizer, max_length=256):
    texts, idxs = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return enc, torch.tensor(idxs, dtype=torch.long)


def prepare_embeddings(model_name, dataset, save_path, 
                       last_pooling=False, model_kwargs={}, tokenizer_kwargs={}):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, 256)
    )

    all_vecs = []
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Encoding")
        for enc, idxs in iterator:
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            if last_pooling:
                embedding = outputs.last_hidden_state[:, -1, :]
            else:
                embedding = outputs.last_hidden_state[:, 0, :]
            all_vecs.append(embedding.cpu())

    all_vecs = torch.cat(all_vecs, dim=0)  # shape (N, hidden)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(all_vecs, save_path)
    print(f"Saved {all_vecs.shape[0]} vectors (dim={all_vecs.shape[1]}) to: {save_path}")

if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.data_path)
    texts = df["text"].astype(str).tolist()
    dataset = TextDataset(texts)

    confs = {
        "bge_conf": {
            "model_name": "BAAI/bge-m3", 
            "dataset": dataset, 
            "save_path": args.save_dir + "/bge-m3-embedding-multi-data.pt",
            "last_pooling":False,
            "model_kwargs":{"torch_dtype": torch.float16}, 
            "tokenizer_kwargs":{}
        },
        "qwen3_0_6B_conf": {
            "model_name": "Qwen/Qwen3-Embedding-0.6B", 
            "dataset": dataset, 
            "save_path": args.save_dir + "/qwen3-0.6B-embedding-multi-data.pt",
            "last_pooling":True,
            "model_kwargs":{"torch_dtype": torch.float16}, 
            "tokenizer_kwargs":{"padding_side": 'left'}
        },
        "qwen3_4B_conf": {
            "model_name": "Qwen/Qwen3-Embedding-4B", 
            "dataset": dataset, 
            "save_path": args.save_dir + "/qwen3-4B-embedding-multi-data.pt",
            "last_pooling":True,
            "model_kwargs":{"torch_dtype": torch.float16}, 
            "tokenizer_kwargs":{"padding_side": 'left'}
        }
    }

    for _, conf in confs.items():
        prepare_embeddings(**conf)

    

