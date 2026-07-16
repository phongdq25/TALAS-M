export CUDA_VISIBLE_DEVICES=0
python ./tools/prepare_embedding.py \
    --data_path './data/train_full.csv' \
    --save_dir './data/teachers_embed_train'