#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2024/09/02 10:40:48
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib
import pandas as pd
from utils import parse_finetune_args,set_random_seed, get_featurizers, SMILESToBigraph, MoleculeDataset,read_dataset
from train import Finetuner
import warnings
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    args,logger = parse_finetune_args()
    
    if args.task in ['classification','regression']:
      for task in args.task_list:
          args.task_list = [task]
          args.task_name = task
          
          # data
          data = read_dataset(args.data_path,target=task,task=args.task)
          logger.info(f"{args.data_path} data loaded, shape: {data.shape}")
          
          if args.additional_features and args.embed_molecular_features == True:
            additional_features = pd.read_csv(args.additional_features)
          else:
            additional_features = None
          
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
          
          logger.info(f"Fine-tuning task: {task}")
          finetuner = Finetuner(dataset=dataset, config=args, logger=logger)
          finetuner.train()     
    else:
      args.task_name = 'multi-task'
      # data
      data = pd.read_csv(args.data_path)
      logger.info(f"{args.data_path} data loaded, shape: {data.shape}")
      
      if args.additional_features:
        additional_features = pd.read_csv(args.additional_features)
      else:
        additional_features = None
      
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
      
      finetuner = Finetuner(dataset=dataset, config=args, logger=logger)
      finetuner.train()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    