from typing import Tuple
import torch
from torch import nn
from torch import optim
import numpy as np
import random

from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from transformers import PreTrainedModel

from argument import parse_args
from model import build_model, build_projection_layer
from dataset import build_dataloader
from transformers import get_scheduler
from trainer import Trainer
from eval import (
    eval_classification_task, 
    eval_pair_task, 
    eval_sts_task
)
from eval_config import (
    dev_clss_tasks,
    dev_pair_tasks,
    dev_sts_tasks,
    test_clss_tasks,
    test_pair_tasks,
    test_sts_tasks
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_optimizer(args, model: PreTrainedModel, 
                    proj_hidden_layers: nn.ModuleList = None) -> Tuple[optim.Optimizer, optim.Optimizer]:
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_edge_or_head = any(keyword in name
                            for keyword in ["embeddings", "pooler", "projector",
                                            "classifier", "head", "layer.0",
                                            "layer.1", "layer.2", "layer.3", 
                                            "layer.4", "layer.10", "layer.11"])

        is_1d_or_norm = param.ndim < 2 or "LayerNorm" in name or "bias" in name

        if not is_edge_or_head and not is_1d_or_norm:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    if not args.adjust_lr_fn:
        opt_muon = optim.Muon(muon_params, lr=args.learning_rate_muon)
    elif args.adjust_lr_fn == "match_rms_adamw":
        opt_muon = optim.Muon(muon_params, lr=args.learning_rate_muon, 
                adjust_lr_fn="match_rms_adamw")
    else:
        raise ValueError(f"Unknown adjust_lr_fn: {args.adjust_lr_fn}")

    opt_adamw = optim.AdamW(adamw_params, lr=args.learning_rate_adamw)
    opt_adamw.add_param_group({"params": proj_hidden_layers.parameters(),
                            "lr": 1e-4, "weight_decay": args.weight_decay})
    
    return opt_muon, opt_adamw

def train(args, trainer, opt_muon, opt_adamw, train_loader):
    num_steps = len(train_loader)
    total_training_steps = num_steps * args.num_train_epochs

    scaler = GradScaler()

    scheduler_adamw = get_scheduler(
        name='cosine_with_min_lr',
        optimizer=opt_adamw,
        num_warmup_steps=int(total_training_steps * args.warmup_ratio),
        num_training_steps=total_training_steps,
        scheduler_specific_kwargs={'min_lr': 5e-5}
    )

    scheduler_muon = get_scheduler(
        name='cosine_with_min_lr',
        optimizer=opt_muon,
        num_warmup_steps=int(total_training_steps * args.warmup_ratio),
        num_training_steps=total_training_steps,
        scheduler_specific_kwargs={'min_lr': 5e-6}
    )

    # Training loop
    for epoch in range(args.num_train_epochs):
        print(('\n' + '%8s' + '%14s' + '%17s' * 2) % ('epoch', 'memory', 'loss', 'student_loss'))
        p_bar = tqdm(train_loader, total=len(train_loader))
        loss_total = 0
        student_loss_total = 0
        step = 0

        for batch in p_bar:
            student_inputs, teacher_embeddings = batch
            teacher_embeddings = teacher_embeddings.to(trainer.student.device)

            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)

            with autocast():
                loss, student_loss = trainer.compute_loss(student_inputs, teacher_embeddings)

            # Backward với Scaler
            scaler.scale(loss).backward()

            # 2. GRADIENT CLIPPING DÀNH CHO AMP
            scaler.unscale_(opt_adamw)
            scaler.unscale_(opt_muon)
            torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(opt_muon)
            scaler.step(opt_adamw)

            scaler.update()

            # 3. UPDATE CẢ 2 SCHEDULER
            scheduler_adamw.step()
            scheduler_muon.step()

            loss_total += loss.item()
            student_loss_total += student_loss.item()
            step += 1

            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
            s = ('%8s' + '%14s' + '%17.5g' * 2) % (f'{epoch + 1}/{args.num_train_epochs}', memory,
                                                    loss_total / step, student_loss_total / step)
            p_bar.set_description(s)

            if torch.isnan(loss):
                print("CẢNH BÁO: Loss bị NaN, ngắt epoch hiện tại!")
                break

        print("Evaluating on validation set...")
        avg_f1 = eval_classification_task(trainer.student, 
                                          trainer.tokenizer, 
                                          args.val_batch_size, 
                                          dev_clss_tasks)
        
        avg_ap = eval_pair_task(trainer.student, 
                                trainer.tokenizer, 
                                args.val_batch_size, 
                                dev_pair_tasks)
        
        avg_spr = eval_sts_task(trainer.student, 
                                trainer.tokenizer, 
                                args.val_batch_size, 
                                dev_sts_tasks)

        print("avg_f1, avg_ap, avg_spr: ", avg_f1, avg_ap, avg_spr)
        print("AVG: ", (avg_f1 + avg_ap + avg_spr) / 3)

        trainer.student.save_pretrained(args.output_dir + f'-epoch{epoch}')

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # build model and projection layers
    model, tokenizer = build_model(args.student_model, args.student_tokenizer, args.device)
    
    # build dataloader
    train_dataset, train_loader = build_dataloader(args.train_data, args.teacher_embedding_path, 
                                    tokenizer, args.passage_max_len, args.batch_size)

    args.teacher_embedding_dimension = train_dataset.teacher_embedding.size(-1)

    proj_hidden_layers = build_projection_layer(model.config.num_hidden_layers, model.config.hidden_size, args.teacher_embedding_dimension, args.device)

    opt_muon, opt_adamw = build_optimizer(args, model, proj_hidden_layers)

    trainer = Trainer(model, tokenizer, proj_hidden_layers, args)

    train(args, trainer, opt_muon, opt_adamw, train_loader)

    # Evaluate on test set

    print("============Evaluating on test set...===============")

    avg_f1 = eval_classification_task(trainer.student, trainer.tokenizer, args.val_batch_size, test_clss_tasks)
    avg_ap = eval_pair_task(trainer.student, trainer.tokenizer, args.val_batch_size, test_pair_tasks)
    avg_spr = eval_sts_task(trainer.student, trainer.tokenizer, args.val_batch_size, test_sts_tasks)

    print("avg_f1, avg_ap, avg_spr: ", avg_f1, avg_ap, avg_spr)
    print("AVG: ", (avg_f1 + avg_ap + avg_spr) / 3)

if __name__ == "__main__":
    main()