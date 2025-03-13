from argparse import ArgumentParser
import argparse
from datetime import datetime
import os
import torch
from .logger import initialize_exp
import pandas as pd
import json

def str2bool(v):
    """
    Convert a string to boolean value.
    
    Args:
        v: Input value, can be boolean or string
        
    Returns:
        bool: Converted boolean value
        
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_finetune_args(parser):
    """
    Add finetune arguments to the argument parser.
    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    Returns:
        None
    """
    
    # device, seed, and logging
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='which device to use')
    parser.add_argument('--seed', type=int, default=2024, help='seed')
    parser.add_argument('--dump_path', type=str, default='dumped', help='path to dump the experiment')
    parser.add_argument('--exp_name', type=str, default='Finetune', help='experiment name')
    parser.add_argument('--exp_id', type=str, default='', help='experiment id')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='log every n steps, default 2')
    parser.add_argument('--eval_every_n_epochs', type=int, default=1, help='evaluate every n epochs, default 1')
    
    # train config    
    parser.add_argument('--batch_size', type=int, default=256, help='batch size, default 256')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, default 100')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default 0.0005')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay, default 0.0001')
    parser.add_argument('--early_stopping_metric', type=str, default=None, help='early stopping metric, default None')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping, default 10')
    parser.add_argument('--disable_tqdm', type=str2bool, default=True, help='disable tqdm, default True')
    parser.add_argument('--normalize', type=str2bool, default=False, help='normalize targets, default True')
    
    
    # dataset and task
    parser.add_argument('--data_path', type=str, default='../dataset/finetune/ocelot/ocelot_clean.csv', help='path to the dataset')
    parser.add_argument('--task', type=str, default='multi-task-regression', choices=['classification', 'regression','multi-task-regression','multi-task-classification' ],help='task type, default multi-task-regression')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers, default 8')
    parser.add_argument('--split_type', type=str, default='random',choices = ['random','scaffold','pre-define'], help='dataset split type, default random')
    parser.add_argument('--valid_size', type=float, default=0.1, help='validation size, default 0.1, only used when split_type is random')
    parser.add_argument('--test_size', type=float, default=0.1, help='test size, default 0.1, only used when split_type is random')
    
    # resume
    parser.add_argument('--resume_from', type=str, default=None, help='path to resume from')
    parser.add_argument('--fine_tune_from', type=str, default=None, help='directory of pre-trained model')
    parser.add_argument('--train_from_scratch', type=str2bool, default=False, help='train from scratch, default False')
    
    # embed molecular features
    parser.add_argument('--embed_molecular_features', type=str2bool, default=True, help='embed molecular features, default True')
    parser.add_argument('--additional_features', type=str, default=None, help='path to the additional features')
    
def add_project_args(parser):
    """
    Add finetune arguments to the argument parser.
    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    Returns:
        None
    """
    
    # device, seed, and logging
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='which device to use')
    parser.add_argument('--seed', type=int, default=2024, help='seed')
    parser.add_argument('--dump_path', type=str, default='dumped', help='path to dump the experiment')
    parser.add_argument('--exp_name', type=str, default='Project', help='experiment name')
    parser.add_argument('--exp_id', type=str, default='project', help='experiment id')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='log every n steps, default 2')
    parser.add_argument('--eval_every_n_epochs', type=int, default=1, help='evaluate every n epochs, default 1')
    
    # train config    
    parser.add_argument('--batch_size', type=int, default=256, help='batch size, default 256')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, default 100')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default 0.0005')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay, default 0.0001')
    parser.add_argument('--early_stopping_metric', type=str, default=None, help='early stopping metric, default None')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping, default 10')
    parser.add_argument('--disable_tqdm', type=str2bool, default=True, help='disable tqdm, default True')
    parser.add_argument('--normalize', type=str2bool, default=False, help='normalize targets, default True')
    
    
    # dataset and task
    parser.add_argument('--data_path', type=str, default='../dataset/ocelot/ocelot.csv', help='path to the dataset')
    parser.add_argument('--task', type=str, default='multi-task-regression', choices=['classification', 'regression','multi-task-regression','multi-task-classification' ],help='task type, default regression')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers, default 8')
    parser.add_argument('--split_type', type=str, default='random',choices = ['random','scaffold','pre-define'], help='dataset split type, default random')
    parser.add_argument('--valid_size', type=float, default=0.1, help='validation size, default 0.1, only used when split_type is random')
    parser.add_argument('--test_size', type=float, default=0.1, help='test size, default 0.1, only used when split_type is random')
    
    # resume
    parser.add_argument('--resume_from', type=str, default=None, help='path to resume from')
    parser.add_argument('--fine_tune_from', type=str, default=None, help='directory of pre-trained model')
    parser.add_argument('--train_from_scratch', type=str2bool, default=False, help='train from scratch, default False')
    
    # fearturizer
    parser.add_argument('--atom_featurizer', type=str, default='canonical', help='atom featurizer, default canonical')
    parser.add_argument('--bond_featurizer', type=str, default='rgcnetype', help='bond featurizer, default rgcnetype')
    
    # embed molecular features
    parser.add_argument('--embed_molecular_features', type=str2bool, default=False, help='embed molecular features, default False')
    parser.add_argument('--additional_features', type=str, default=None, help='path to the additional features')


def add_predict_args(parser):
    """
    Add predict arguments to the argument parser.
    Args:
        parser (argparse.ArgumentParser): The argument parser object.
    Returns:
        None
    """
    
    # device, seed, and logging
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='which device to use')
    parser.add_argument('--seed', type=int, default=2024, help='seed')
    parser.add_argument('--dump_path', type=str, default='dumped', help='path to dump the experiment')
    parser.add_argument('--exp_name', type=str, default='Predict', help='experiment name')
    parser.add_argument('--exp_id', type=str, default='predict', help='experiment id')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='log every n steps, default 2')
    parser.add_argument('--eval_every_n_epochs', type=int, default=1, help='evaluate every n epochs, default 1')
    
    # train config    
    parser.add_argument('--batch_size', type=int, default=256, help='batch size, default 256')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, default 100')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default 0.0005')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay, default 0.0001')
    parser.add_argument('--early_stopping_metric', type=str, default=None, help='early stopping metric, default None')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping, default 10')
    parser.add_argument('--disable_tqdm', type=str2bool, default=True, help='disable tqdm, default True')
    parser.add_argument('--normalize', type=str2bool, default=False, help='normalize targets, default True')
    
    
    # dataset and task
    parser.add_argument('--data_path', type=str, default='../dataset/finetune/ocelot/ocelot_clean.csv', help='path to the dataset')
    parser.add_argument('--task', type=str, default='regression', choices=['classification', 'regression','multi-task-regression','multi-task-classification' ],help='task type, default multi-task-regression')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers, default 8')
    parser.add_argument('--split_type', type=str, default='random',choices = ['random','scaffold','pre-define'], help='dataset split type, default random')
    parser.add_argument('--valid_size', type=float, default=0.1, help='validation size, default 0.1, only used when split_type is random')
    parser.add_argument('--test_size', type=float, default=0.1, help='test size, default 0.1, only used when split_type is random')
    
    # resume
    parser.add_argument('--resume_from', type=str, default=None, help='path to resume from')
    parser.add_argument('--fine_tune_from', type=str, default=None, help='directory of pre-trained model')
    parser.add_argument('--train_from_scratch', type=str2bool, default=False, help='train from scratch, default False')
    
    # embed molecular features
    parser.add_argument('--embed_molecular_features', type=str2bool, default=True, help='embed molecular features, default True')
    parser.add_argument('--additional_features', type=str, default=None, help='path to the additional features')
    
    # predict data path
    parser.add_argument('--predict_data_path', type=str, default=None, help='path to the predict data')

def parse_finetune_args():
    """
    Parse the arguments for pretraining
    """
    
    parser = ArgumentParser()
    add_finetune_args(parser)
    args = parser.parse_args()
    args,logger = modify_finetune_args(args)
    return args,logger

def parse_project_args():
    """
    Parse the arguments for pretraining
    """
    
    parser = ArgumentParser()
    add_project_args(parser)
    args = parser.parse_args()
    args,logger = modify_finetune_args(args)
    return args,logger

def parse_predict_args():
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    args,logger = modify_finetune_args(args)
    return args,logger

def modify_finetune_args(args):
    """
    Modify the pretraining arguments
    """

    # train device
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    
    # logger and dump path initialization
    logger,exp_folder = initialize_exp(args)
    # args.logger = logger
    
    # check dump path
    # if os.path.exists(args.dump_path):
    #     # num +1
    #     args.dump_path = os.path.join(args.dump_path, f'{args.exp_name}-{args.exp_id}')
    
    args.dump_path = exp_folder
    
    # check dataset 
    if args.data_path is None:
        raise ValueError('data path is None')
    if not os.path.exists(args.data_path):
        raise ValueError(f'{args.data_path} not exists')
    
    # check task_list
    # read csv columns
    columns = pd.read_csv(args.data_path, nrows=0).columns.tolist()
    if 'smiles' not in columns:
        raise ValueError('smiles not in the dataset')
    
    task_list = [col for col in columns if col not in ['smiles','split']]        

    if len(task_list) == 0:
        raise ValueError('No task detected')
    if len(task_list) == 1:
        logger.info(f'Single task detected: {task_list}')
    else:
        logger.info(f'Multiple tasks detected: {task_list}')
    
    args.task_list = task_list
    
    # check fine_tune_from
    if args.fine_tune_from is None or not os.path.exists(args.fine_tune_from):
        raise ValueError('fine_tune_from is None or not exists')
    else:
        logger.info(f'Fine-tuning from {args.fine_tune_from}')
    
    # check model config
    config_path = os.path.join(args.fine_tune_from, 'checkpoints','config.json')
    model_config_path = os.path.join(args.fine_tune_from, 'checkpoints', 'model_config.json')
    if not os.path.exists(config_path) or not os.path.exists(model_config_path):
        raise ValueError('config.json or model_config.json not exists')
    
    args.model = json.load(open(config_path, 'r'))['model']
    args.model_config = json.load(open(model_config_path, 'r'))
    logger.info("*"*100)
    logger.info(f'Ckpt model: {args.model}, model config: {args.model_config}')
    logger.info("*"*100)
    
    # check atom_featurizer and bond_featurizer
    args.atom_featurizer = json.load(open(config_path, 'r'))['atom_featurizer']
    args.bond_featurizer = json.load(open(config_path, 'r'))['bond_featurizer']
    logger.info(f'Atom featurizer: {args.atom_featurizer}, bond featurizer: {args.bond_featurizer}')
    
    # check task list
    if args.task not in ['classification', 'regression','multi-task-regression','multi-task-classification']:
        raise ValueError(f'{args.task} not supported')

    # check early stopping
    if args.task in ['classification', 'multi-task-classification'] and args.early_stopping_metric is None:
        args.early_stopping_metric = 'accuracy'
        args.early_stopping_mode = 'higher'
        logger.info(f'Early stopping metric: {args.early_stopping_metric}, mode: {args.early_stopping_mode}')
        
    elif args.task in ['regression', 'multi-task-regression'] and args.early_stopping_metric is None:
        args.early_stopping_metric = 'r2'
        args.early_stopping_mode = 'higher'
        logger.info(f'Early stopping metric: {args.early_stopping_metric}, mode: {args.early_stopping_mode}')
    
    if args.early_stopping_metric is not None:
        if args.early_stopping_metric in ['accuracy', 'r2']:
            args.early_stopping_mode = 'higher'
        elif args.early_stopping_metric in ['mae', 'rmse']:
            args.early_stopping_mode = 'lower'
        
        logger.info(f'Early stopping metric: {args.early_stopping_metric}, mode: {args.early_stopping_mode}')

    # check molecular_features_dim
    if args.embed_molecular_features:
        if os.path.exists(args.additional_features):
            # 只需要先读取形状即可,不需要读取数据,不要占用内存
            args.molecular_features_dim = pd.read_csv(args.additional_features).shape[1]-1 # -1 for smiles column
        else:
            raise ValueError(f'{args.additional_features} not exists')
        if args.additional_features.endswith('qm9_sub_descriptors_processed.csv'):
            args.molecular_features_dim = 653
    else:
        args.molecular_features_dim = 0

    return args,logger


