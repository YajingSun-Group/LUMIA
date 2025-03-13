#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2024/08/27 15:03:19
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib
import pandas as pd
from utils import parse_pretrain_args,set_random_seed, \
                  CanonicalAtomFeaturizer, RGCNETypeFeaturizer, \
                  get_featurizers, SMILESToBigraph, \
                    MoleculeDataset

from train import Pretrainer

def main():
    args,logger = parse_pretrain_args()
    
    # data loading
    data = pd.read_csv(args.dataset_path)
    logger.info(f"{args.dataset} data loaded, shape: {data.shape}")    
    
    # set random seed
    set_random_seed(args.seed)
    
    # set dataset and dataloader
    atom_featurizer, bond_featurizer = get_featurizers(args)
    smiles_to_g = SMILESToBigraph(node_featurizer=atom_featurizer,
                                  edge_featurizer=bond_featurizer)
    
    dataset = MoleculeDataset(data=data,
                                 smiles_to_graph=smiles_to_g,
                                 args=args)
    
    pretrainer = Pretrainer(dataset=dataset, config=args, logger=logger)
    pretrainer.train()
    
if __name__ == '__main__':
    main()
    
    
    
