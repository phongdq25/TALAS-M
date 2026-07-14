import argparse



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train talas model"
    )

    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="data train",
    )

    parser.add_argument(
        "--student_model",
        type=str,
        required=True,
        help="Tên hoặc đường dẫn model",
    )

    parser.add_argument(
        "--student_tokenizer",
        type=str,
        required=True,
        help="Thư mục lưu checkpoint",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size trên mỗi GPU",
    )

    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=64,
        help="Batch size trên mỗi GPU cho validation",
    )
    
    parser.add_argument(
        "--passage_max_len",
        type=int,
        default=256,
        help="Chiều dài tối đa của passage",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Số epoch",
    )

    parser.add_argument(
        "--learning_rate_muon",
        type=float,
        default=1e-3,
        help="Learning rate for MUON",
    )

    parser.add_argument(
        "--learning_rate_adamw",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05, 
        help="Warmup ratio",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature",
    )

    parser.add_argument(
        "--adjust_lr_fn",
        type=str,
        default=None,
        choices=["match_rms_adamw"],
        help="adjust learning rate for MUON",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Thư mục lưu checkpoint",
    )

    parser.add_argument(
        "--teacher_embedding_path",
        type=str,
        required=True,
        help="teacher embedding",
    )

    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.001,
        help="weight for simcse loss",
    )

    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.75,
        help="weight for TAMD loss",
    )

    parser.add_argument(
        "--lambda3",
        type=float,
        default=1.0,
        help="weight for LASD loss",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()