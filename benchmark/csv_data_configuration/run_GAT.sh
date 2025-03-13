python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_0.csv \
    -p ocelot_results/GAT/cv_0 \
    -mo GAT \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo


python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_1.csv \
    -p ocelot_results/GAT/cv_1 \
    -mo GAT \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo

python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_2.csv \
    -p ocelot_results/GAT/cv_2 \
    -mo GAT \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo

python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_3.csv \
    -p ocelot_results/GAT/cv_3 \
    -mo GAT \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo

python regression_train.py \
    -c /home/qianzhang/MyProject/LUMIA/dataset/finetune/ocelot/5_fold_cross_validation/ocelot_clean_5fold_4.csv \
    -p ocelot_results/GAT/cv_4 \
    -mo GAT \
    --device cuda:1 \
    -sc smiles \
    -n 100 \
    -s predefine \
    -split-column split \
    -t vie,aie,vea,aea,hl,s0s1,s0t1,hr,cr2,cr1,er,ar1,ar2,lumo,homo

