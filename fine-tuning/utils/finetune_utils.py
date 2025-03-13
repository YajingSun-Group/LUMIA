

# here put the import lib
import random
import torch
import dgl
import numpy as np
from .featurizer import CanonicalAtomFeaturizer, RGCNETypeFeaturizer, CanonicalBondFeaturizer


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
    
    if args.bond_featurizer == 'rgcnetype' and (args.model == 'RGCN' or args.model == 'rgcn'):
        bond_featurizer = RGCNETypeFeaturizer(edge_data_field=bond_data_field)
        args.model = 'RGCN'
    
    elif args.bond_featurizer == 'canonical':
        bond_featurizer = CanonicalBondFeaturizer(edge_data_field=bond_data_field)
        
    return atom_featurizer, bond_featurizer


def count_model_parameters(model):
    """Count the parameters of a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        non_trainable_params: Number of non-trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print('='*50)
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    print(f'Non-trainable Parameters: {non_trainable_params:,}')
    print('='*50)
    
    return total_params, trainable_params, non_trainable_params