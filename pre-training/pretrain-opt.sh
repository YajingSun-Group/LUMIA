python pretrain-opt.py \
    --device cuda:0 \
    --seed 0 \
    --dump_path dumped \
    --exp_name pretrain-opt \
    --exp_id encoder-RGCN  \
    --log_every_n_steps 10 \
    --eval_every_n_epochs 1 \
    --model RGCN \
    --model_config ./train/model/configs/RGCN.json \
    --batch_size 512 \
    --epochs 100 \
    --lr 0.0005 \
    --weight_decay 0.00001 \
    --warmup 10 \
    --dataset ../dataset/pretrain/200k.csv \
    --num_workers 8 \
    --valid_size 0.05 \
    --lambda_1 0.5 \
    --lambda_2 0.5 \
    --temperature 0.1 \
    --use_cosine_similarity True \
    --mask_edge True \
    --mask_substituent True \
    --mask_rate 0.25 \
    --atom_featurizer canonical \
    --bond_featurizer rgcnetype \
