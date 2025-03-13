
python predict.py \
    --device cuda:0 \
    --seed 2034 \
    --dump_path ./dumped \
    --exp_name predict-last \
    --exp_id cross_validation_0 \
    --log_every_n_steps 10 \
    --eval_every_n_epochs 1 \
    --batch_size 1 \
    --epochs 2000 \
    --lr 0.0001 \
    --weight_decay 0.00001 \
    --early_stopping_metric mae \
    --patience 40 \
    --disable_tqdm False \
    --normalize True \
    --data_path ../dataset/finetune/ocelot/5_fold_cross_validation/cv_split/cross_validation_0/hr.csv \
    --task regression \
    --num_workers 4 \
    --split_type random \
    --valid_size 0.2 \
    --fine_tune_from ../pre-training/dumped/1205-pretrain-full/encoder-RGCN \
    --resume_from /home/qianzhang/MyProject/LUMIA/fine-tuning/dumped/1213-finetune-ocelot-no-embed/cross_validation_0/hr \
    --embed_molecular_features False\
    --predict_data_path ./last.csv \


