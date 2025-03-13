#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   project.py
@Time    :   2024/09/06 17:20:42
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib
# here put the import lib
import pandas as pd
from utils import parse_finetune_args,parse_project_args,set_random_seed, get_featurizers, SMILESToBigraph, MoleculeDataset
from train import Finetuner
import warnings
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    args,logger = parse_project_args()
    args.task_name = 'multi-task'
    if args.additional_features:
        additional_features = pd.read_csv(args.additional_features)
    else:
        additional_features = None
    # data
    data = pd.read_csv(args.data_path)
    logger.info(f"{args.data_path} data loaded, shape: {data.shape}")
    
    # set random seed
    set_random_seed(args.seed)
    
    # set dataset and dataloader
    atom_featurizer, bond_featurizer = get_featurizers(args)
    smiles_to_g = SMILESToBigraph(node_featurizer=atom_featurizer,
                                  edge_featurizer=bond_featurizer,
                                  )
    
    dataset = MoleculeDataset(data=data,
                              additional_features=additional_features,
                              smiles_to_graph=smiles_to_g,
                              args=args)
    
    train_loader, valid_loader, test_loader = dataset.get_data_loaders()
    
    
    # if args.task in ['classification','regression']:
    #   for task in args.task_list[:1]:
    #       args.task_list = [task]
    #       args.task_name = task
    #       logger.info(f"Fine-tuning task: {task}")
    #       finetuner = Finetuner(dataset=dataset, config=args, logger=logger)
    #       finetuner.build_model()     
    # else:
    finetuner = Finetuner(dataset=dataset, config=args, logger=logger)
    finetuner.build_model()
      
    
    # project
    finetuner.project(train_loader,tag='train')
    finetuner.project(valid_loader,tag='valid')
    finetuner.project(test_loader,tag='test')
    logger.info("Project finished!")
    
if __name__ == '__main__':
    main()
    
    
    