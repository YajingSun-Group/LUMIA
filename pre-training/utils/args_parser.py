from argparse import ArgumentParser
from datetime import datetime
import os
import torch
from .logger import initialize_exp

def add_pretrain_args(parser):
    """
    Add pretraining arguments to the argument parser.
    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    Returns:
        None
    """
    
    # device, seed, and logging config
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computing device to use (cuda:0, cpu, etc)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--dump_path', type=str, default='dumped', help='Directory to save model checkpoints')
    parser.add_argument('--exp_name', type=str, default='pretrain', help='Name of the experiment')
    parser.add_argument('--exp_id', type=str, default='0', help='Unique identifier for the experiment')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='Number of steps between logging, default 2')
    parser.add_argument('--eval_every_n_epochs', type=int, default=1, help='Number of epochs between evaluations, default 1')
    
    # model config
    parser.add_argument('--model', type=str, default='RGCN', help='model type', choices=['RGCN', 'GCN', 'GIN', 'GAT', 'MPNN'])
    parser.add_argument('--model_config', type=str, default=None, help='model config')

    # train config    
    parser.add_argument('--batch_size', type=int, default=256, help='batch size, default 256')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, default 100')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate, default 0.0005')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay, default 0.0001')
    parser.add_argument('--warmup', type=int, default=10, help='warmup epochs, default 10')
    
    # dataset config
    parser.add_argument('--dataset', type=str, default='test', help='path of the dataset for self-supervised pretraining. (default: zinc_standard_agent)')
    parser.add_argument('--num_workers', type=int, default=20, help='number of workers, default 8')
    parser.add_argument('--valid_size', type=float, default=0.05, help='validation size, default 0.05')
    parser.add_argument('--graph_data_path', type=str, default=None, help='path of the graph data for self-supervised pretraining')
    
    # resume config
    parser.add_argument('--resume_from', type=str, default='', help='path to resume from')
    
    # loss config
    parser.add_argument('--lambda_1', type=float, default=0.5, help='lambda_1')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='lambda_2')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature')
    parser.add_argument('--use_cosine_similarity', type=bool, default=True, help='use cosine similarity')    
    
    # pretrain strategy config
    parser.add_argument('--mask_edge', type=bool, default=True, help='Whether to perform edge masking during pretraining')
    parser.add_argument('--mask_substituent', type=bool, default=True, help='Whether to perform substructure masking during pretraining')
    parser.add_argument('--mask_rate', type=float, default=0.25, help='Ratio of elements to mask during pretraining')
    
    # fearturizer config
    parser.add_argument('--atom_featurizer', type=str, default='canonical', help='atom featurizer, default canonical')
    parser.add_argument('--bond_featurizer', type=str, default='rgcnetype', help='bond featurizer, default rgcnetype')
    


def parse_pretrain_args():
    """
    Parse the arguments for pretraining
    """
    
    parser = ArgumentParser()
    add_pretrain_args(parser)
    args = parser.parse_args()
    args,logger = modify_pretrain_args(args)
    return args,logger

def modify_pretrain_args(args):
    """
    Modify the pretraining arguments
    """

    # train devicez
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    
    # logger and dump path initialization
    logger,exp_folder = initialize_exp(args)
    # args.logger = logger
    args.dump_path = exp_folder
    
    # dataset 
    if args.dataset == 'test':
        args.dataset_path = 'dataset/zinc15-250K/zinc15_200K.csv'
    elif args.dataset == 'zinc15_2M':
        args.dataset_path = 'dataset/zinc15-2M-final/zinc15_2M.csv'
    else:
        args.dataset_path = args.dataset
        
    # model config
    if args.model == 'rgcn' and args.model_config is None:
        args.model_config = 'train/model/configs/rgcn.json'
    
    return args,logger
