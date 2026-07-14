export CUDA_VISIBLE_DEVICES=0
python eval.py \
    --model_name 'bert-base-uncased' \
    --tokenizer 'bert-base-uncased' \
    --batch_size 64 \
    --device 'cuda' \
    --seed 42