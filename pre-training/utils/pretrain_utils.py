#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   seed.py
@Time    :   2024/08/26 15:22:27
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib
import random
import torch
import dgl
import numpy as np
from .featurizer import CanonicalAtomFeaturizer, RGCNETypeFeaturizer,CanonicalBondFeaturizer
import os
from datetime import datetime
from torch.nn.parallel import DataParallel, DistributedDataParallel

def set_random_seed(seed):
    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_featurizers(args, atom_data_field = 'node', bond_data_field = 'edge'):
    if args.atom_featurizer == 'canonical':
        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field=atom_data_field)
    
    if args.bond_featurizer == 'rgcnetype' and args.model == 'RGCN':
        bond_featurizer = RGCNETypeFeaturizer(edge_data_field=bond_data_field)
    elif args.bond_featurizer == 'canonical':
        bond_featurizer = CanonicalBondFeaturizer(bond_data_field=bond_data_field)
    return atom_featurizer, bond_featurizer


def snapshot(model, epoch, save_path, name):
    """
    Saving model w/ its params.
        Get rid of the ONNX Protocal.
    F-string feature new in Python 3.6+ is used.
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    save_path = os.path.join(save_path, f'{name}_{type(model).__name__}_{timestamp}_{epoch}th_epoch.pkl')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
        


def count_model_parameters(model):
    """Count parameters of the RGCN model by components.
    
    Args:
        model: RGCN model
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        non_trainable_params: Number of non-trainable parameters
    """
    total_params = 0
    trainable_params = 0
    
    print('\n' + '='*50)
    print('Model Parameters Summary:')
    
    # 1. RGCN Layers
    print('\nRGCN Layers:')
    layer_count = 0
    rgcn_total = 0
    for name, module in model.named_modules():
        if 'RGCNLayer' in module.__class__.__name__:
            layer_count += 1
            layer_total = 0
            
            # TypedLinear参数
            typed_linear_params = sum(p.numel() for p in module.graph_conv_layer.linear_r.parameters())
            # 残差连接参数
            res_params = sum(p.numel() for p in module.res_connection.parameters())
            # 批归一化参数
            bn_params = sum(p.numel() for p in module.bn_layer.parameters())
            
            layer_total = typed_linear_params + res_params + bn_params
            rgcn_total += layer_total
            
            print(f'\nRGCN Layer {layer_count}:')
            print(f'  - TypedLinear: {typed_linear_params:,} parameters')
            print(f'  - Residual Connection: {res_params:,} parameters')
            print(f'  - BatchNorm: {bn_params:,} parameters')
            print(f'  Layer total: {layer_total:,} parameters')
    
    print(f'\nTotal RGCN Layers: {rgcn_total:,} parameters')
    total_params += rgcn_total
    
    # 2. Feature Linear Layer
    feat_lin_params = sum(p.numel() for p in model.feat_lin.parameters())
    print(f'\nFeature Linear Layer: {feat_lin_params:,} parameters')
    total_params += feat_lin_params
    
    # 3. Output Linear Layers
    out_lin_params = sum(p.numel() for p in model.out_lin.parameters())
    print(f'\nOutput Linear Layers: {out_lin_params:,} parameters')
    total_params += out_lin_params
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print('\n' + '='*50)
    print('Overall Model Summary:')
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    print(f'Non-trainable Parameters: {non_trainable_params:,}')
    print('='*50)
    
    return total_params, trainable_params, non_trainable_params
