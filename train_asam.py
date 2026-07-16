from typing import Tuple
import torch
from torch import nn
from torch import optim
import numpy as np
import random

from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from transformers import PreTrainedModel
from pytorch_optimizer import SAM

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
    
    base_optimizer = optim.AdamW

    optimizer = SAM(
        [
            {"params": list(model.parameters()), "lr": args.learning_rate_adamw, 'weight_decay': args.weight_decay},
            {"params": list(proj_hidden_layers.parameters()),  "lr": 1e-4, 'weight_decay': args.weight_decay},
        ],
        base_optimizer,
        rho=0.05,
        adaptive=True,
    )
    return optimizer


def train(args, trainer, opt_sam, train_loader):
    num_steps = len(train_loader)
    total_training_steps = num_steps * args.num_train_epochs

    base_opt = opt_sam.base_optimizer

    scaler = GradScaler()

    scheduler_adamw = get_scheduler(
        name='cosine_with_min_lr',
        optimizer=base_opt,
        num_warmup_steps=int(total_training_steps * args.warmup_ratio),
        num_training_steps=total_training_steps,
        scheduler_specific_kwargs={'min_lr': 5e-6}
    )

    for epoch in range(args.num_train_epochs):
        print(("\n" + "%8s" + "%14s" + "%17s" * 2) % ("epoch", "memory", "loss", "student_loss"))

        p_bar = tqdm(train_loader, total=len(train_loader))

        loss_total = 0.0
        student_loss_total = 0.0
        step = 0

        trainer.student.train()

        for batch in p_bar:
            student_inputs, teacher_embeddings = batch
            teacher_embeddings = teacher_embeddings.to(trainer.student.device)

            # =========================
            # SAM step 1: forward-backward để tìm perturbation
            # =========================
            opt_sam.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, student_loss = trainer.compute_loss(
                    student_inputs,
                    teacher_embeddings,
                )

            if torch.isnan(loss):
                print("WARNING: Loss NaN at SAM step 1, break!")
                break

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainer.student.parameters(),
                max_norm=1.0,
            )

            opt_sam.first_step(zero_grad=True)

            # =========================
            # SAM step 2: forward-backward tại weight đã perturb
            # =========================
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_2, student_loss_2 = trainer.compute_loss(
                    student_inputs,
                    teacher_embeddings,
                )

            if torch.isnan(loss_2):
                print("WARNING: Loss NaN at SAM step 2, break!")
                break

            loss_2.backward()

            torch.nn.utils.clip_grad_norm_(
                trainer.student.parameters(),
                max_norm=1.0,
            )

            opt_sam.second_step(zero_grad=True)

            scheduler_adamw.step()

            loss_total += loss_2.item()
            student_loss_total += student_loss_2.item()
            step += 1

            memory = f"{torch.cuda.memory_reserved() / 1E9:.4g}G"
            s = ("%8s" + "%14s" + "%17.5g" * 2) % (
                f"{epoch + 1}/{args.num_train_epochs}",
                memory,
                loss_total / step,
                student_loss_total / step,
            )
            p_bar.set_description(s)

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

    opt_sam = build_optimizer(args, model, proj_hidden_layers)

    trainer = Trainer(model, tokenizer, proj_hidden_layers, args)

    train(args, trainer, opt_sam, train_loader)

    # Evaluate on test set

    print("============Evaluating on test set...===============")

    avg_f1 = eval_classification_task(trainer.student, trainer.tokenizer, args.val_batch_size, test_clss_tasks)
    avg_ap = eval_pair_task(trainer.student, trainer.tokenizer, args.val_batch_size, test_pair_tasks)
    avg_spr = eval_sts_task(trainer.student, trainer.tokenizer, args.val_batch_size, test_sts_tasks)

    print("avg_f1, avg_ap, avg_spr: ", avg_f1, avg_ap, avg_spr)
    print("AVG: ", (avg_f1 + avg_ap + avg_spr) / 3)

if __name__ == "__main__":
    main()