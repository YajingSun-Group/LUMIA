python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_0.csv \
    -p ocelot_results/GCN/cv_0 \
    -mo GCN \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo

python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_1.csv \
    -p ocelot_results/GCN/cv_1 \
    -mo GCN \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo


python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_2.csv \
    -p ocelot_results/GCN/cv_2 \
    -mo GCN \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo


python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_3.csv \
    -p ocelot_results/GCN/cv_3 \
    -mo GCN \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo


python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_4.csv \
    -p ocelot_results/GCN/cv_4 \
    -mo GCN \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo