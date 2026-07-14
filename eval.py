import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from trainer import get_embedding
from eval_data import STSDataset, ClasssifyDataset, PairDataset
from eval_config import test_clss_tasks, test_pair_tasks, test_sts_tasks


def eval_sts(model, eval_loader):
    preds, labels = [], []
    device = model.device

    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                input_ids2 = batch["input_ids2"].to(device)
                attn2 = batch["attention_mask2"].to(device)
                label = batch["labels"]


                out1 = model(input_ids=input_ids1, attention_mask=attn1)
                out2 = model(input_ids=input_ids2, attention_mask=attn2)

                emb1 = get_embedding(out1.last_hidden_state, attn1)
                emb2 = get_embedding(out2.last_hidden_state, attn2)

                # cosine similarity
                sim = F.cosine_similarity(emb1, emb2)
                score = (sim + 1) * 2.5  # scale [-1,1] -> [0,5]

                preds.extend(score.cpu().numpy())
                labels.extend(label.numpy())

    spearman_corr, _ = spearmanr(preds, labels)
    print(f"Spearman: {spearman_corr:.4f}")

    return spearman_corr

def eval_sts_task(model, tokenizer, batch_size: int, path_list: list[str]):
    model.eval()
    print('eval_sts_task')
    sum_spearman_corr = 0
    for path in path_list:
        print(path)
        eval_dataset = STSDataset(path)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: eval_dataset.collate_fn(x, tokenizer)
        )
        spearman_corr = eval_sts(model, eval_loader)
        sum_spearman_corr += spearman_corr

    model.train()

    return sum_spearman_corr / len(path_list)


def eval_cls(model, eval_loader):
    preds, labels = [], []
    device = model.device

    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                label = batch["labels"]

                out1 = model(input_ids=input_ids1, attention_mask=attn1)
                emb1 = get_embedding(out1.last_hidden_state, attn1)

                preds.extend(emb1.cpu().numpy())
                labels.extend(label.numpy())

    return preds, labels

def eval_classification_task(model, tokenizer, 
                             batch_size: int, path_list: list[tuple[str, str]]):
    model.eval()
    print('classifier')
    sum_f1 = 0

    for train_path, dev_path in path_list:
        print(dev_path)
        eval_dataset = ClasssifyDataset(dev_path)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: eval_dataset.collate_fn(x, tokenizer)
        )

        train_dataset = ClasssifyDataset(train_path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda x: train_dataset.collate_fn(x, tokenizer)
        )

        X_train, y_train = eval_cls(model, train_loader)
        X_test, y_test = eval_cls(model, eval_loader)

        clf = LogisticRegression(
            random_state=42,
            n_jobs=1,
            max_iter=200,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        scores = {}
        accuracy = accuracy_score(y_test, y_pred)
        scores["accuracy"] = accuracy
        f1 = f1_score(y_test, y_pred, average="macro")
        scores["f1"] = f1
        print(scores)
        sum_f1 += f1

    model.train()

    return sum_f1 / len(path_list)


def eval_pair(model, eval_loader):
    preds, labels = [], []
    device = model.device

    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                input_ids2 = batch["input_ids2"].to(device)
                attn2 = batch["attention_mask2"].to(device)
                label = batch["labels"]


                out1 = model(input_ids=input_ids1, attention_mask=attn1)
                out2 = model(input_ids=input_ids2, attention_mask=attn2)

                emb1 = get_embedding(out1.last_hidden_state, attn1)
                emb2 = get_embedding(out2.last_hidden_state, attn2)

                # cosine similarity
                sim = F.cosine_similarity(emb1, emb2)
                score = (sim + 1) / 2

                preds.extend(score.cpu().numpy())
                labels.extend(label.numpy())

    metric = get_metric_pair_classification(preds, labels)
    print(metric)

    return metric

def get_metric_pair_classification(scores, labels):
    best_acc, best_thr = 0, 0
    for thr in np.linspace(0, 1, 200):
        preds = (scores >= thr).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    preds = (scores >= best_thr).astype(int)
    return {
        "best_threshold": best_thr,
        "accuracy": best_acc,
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "average_precision": average_precision_score(labels, scores)
    }

def eval_pair_task(model, tokenizer, batch_size, path_list):
    model.eval()
    print('eval_pair_task')
    sum_average_precision = 0
    for path in path_list:
        print(path)
        eval_dataset = PairDataset(path)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: eval_dataset.collate_fn(x, tokenizer)
        )
        metric = eval_pair(model, eval_loader)
        sum_average_precision += metric['average_precision']

    model.train()
    return sum_average_precision / len(path_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval args")

    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name or path",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer name or path",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Eval batch size",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Start eval model: {args.model_name}")

    model = AutoModel.from_pretrained(args.model_name, 
                                      device_map=args.device, 
                                      output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)


    avg_f1 = eval_classification_task(model, tokenizer, args.batch_size, test_clss_tasks)
    avg_ap = eval_pair_task(model, tokenizer, args.batch_size, test_pair_tasks)
    avg_spr = eval_sts_task(model, tokenizer, args.batch_size, test_sts_tasks)

    print("avg_f1, avg_ap, avg_spr: ", avg_f1, avg_ap, avg_spr)
    print("AVG: ", (avg_f1 + avg_ap + avg_spr) / 3)