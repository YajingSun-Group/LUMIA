from collections import defaultdict
import csv
from functools import partial
import pandas as pd

from sklearn.model_selection import train_test_split
# from .structure_decomposition import get_leaves
from .structure_analyzer import get_structure
from .ifg import get_fgs
import torch

import numpy as np
# import dgl.backend as F
import random
import dgl
# from dgllife.utils import ScaffoldSplitter, RandomSplitter
from torch.utils.data import DataLoader
from networkx.algorithms.components import node_connected_component
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
import networkx as nx
from rdkit import Chem
import signal
from mordred import Calculator, descriptors
from rdkit import Chem


class MoleculeDataLoader(object):
    """
    adapted from https://lifesci.dgl.ai/_modules/dgllife/data/csv_dataset.html#MoleculeCSVDataset
    used for pretrain_masking(task=masking) and pretrain_supervised(task=supervised) task.
    """

    def __init__(self, data, smiles_to_graph,args,additional_features=None,transform=None):
        self.smiles = data['smiles'].tolist()
        self.targets = data[args.task_list].values
        self.smiles_to_graph = smiles_to_graph
        self.args = args
        self.transform = transform
        
        # 预处理 additional_features，创建快速查找字典
        if additional_features is not None:
            # 将 DataFrame 转换为字典，key 是 SMILES，value 是特征向量
            self.feature_dict = dict(zip(
                additional_features['smiles'],
                additional_features.iloc[:, :-1].values  # 除了 smiles 列的所有值,smiles 列是最后一列
            ))
        else:
            self.feature_dict = None
        
    def __getitem__(self, item):
        s = self.smiles[item]
        target = self.targets[item]

        mol = Chem.MolFromSmiles(s)
        g = self.smiles_to_graph(s)
        
        # add smsk attribute in graph for weight and sum readout
        g.ndata['smask'] = torch.tensor(construct_smask([], g.number_of_nodes())).float()
        
        # 使用字典快速查找分子特征，但不移动到 GPU
        if self.feature_dict is not None:
            try:
                mol_features = self.feature_dict[s]
                # 确保特征是数值类型
                # print(mol_features.shape)
                mol_features = np.array(mol_features, dtype=np.float32)  # 显式转换为float32
                setattr(g, 'molecular_features', torch.tensor(mol_features).float())
            except KeyError:
                raise ValueError(f"SMILES {s} not found in additional features")
        # else:
        #     setattr(g, 'molecular_features', None)
        
        return s, g, target

    def __len__(self):
        return len(self.smiles)

class MoleculeTestDataLoader(object):
    def __init__(self, data, smiles_to_graph,args,additional_features=None,transform=None):
        self.smiles = data['smiles'].tolist()
        self.smiles_to_graph = smiles_to_graph
        self.args = args
        self.transform = transform
        
        # 预处理 additional_features，创建快速查找字典
        if additional_features is not None:
            # 将 DataFrame 转换为字典，key 是 SMILES，value 是特征向量
            self.feature_dict = dict(zip(
                additional_features['smiles'],
                additional_features.iloc[:, :-1].values  # 除了 smiles 列的所有值,smiles 列是最后一列
            ))
        else:
            self.feature_dict = None
        
    def __getitem__(self, item):
        s = self.smiles[item]

        mol = Chem.MolFromSmiles(s)
        g = self.smiles_to_graph(s)
        
        # add smsk attribute in graph for weight and sum readout
        g.ndata['smask'] = torch.tensor(construct_smask([], g.number_of_nodes())).float()
        
        # 使用字典快速查找分子特征，但不移动到 GPU
        if self.feature_dict is not None:
            try:
                mol_features = self.feature_dict[s]
                # 确保特征是数值类型
                # print(mol_features.shape)
                mol_features = np.array(mol_features, dtype=np.float32)  # 显式转换为float32
                setattr(g, 'molecular_features', torch.tensor(mol_features).float())
            except KeyError:
                raise ValueError(f"SMILES {s} not found in additional features")
        # else:
        #     setattr(g, 'molecular_features', None)
        
        return s, g

    def __len__(self):
        return len(self.smiles)

class MoleculeDataset(object):
    def __init__(self, data,smiles_to_graph,args,additional_features=None):
        super(MoleculeDataset, self).__init__()
        # self.data = data
        self.data = data
        
        self.split_type = args.split_type
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.valid_size = args.valid_size
        self.test_size = args.test_size
        
        self.smiles_to_g = smiles_to_graph
        self.args = args
        
        self.additional_features = additional_features
    
    def get_full_loader(self):
        """Get a data loader for the full dataset
        
        Returns
        -------
        DataLoader
            A PyTorch DataLoader containing the full dataset with:
            - No shuffling
            - No dropping of last batch
            - Batch size and workers from args
            - Custom collate function for molecules
        """
        all_data = MoleculeTestDataLoader(self.data,
                                  smiles_to_graph=self.smiles_to_g,
                                  additional_features=self.additional_features,
                                  args=self.args,
                                  transform=None)
        
        return DataLoader(all_data,
                          batch_size=self.batch_size,
                          collate_fn=collate_fn_predict,
                          num_workers=self.num_workers,
                          drop_last=False,
                          shuffle=False)
        
    def get_data_loaders(self):
        
        if self.split_type == 'random':
            return self.get_random_split()
        elif self.split_type == 'scaffold':
            return self.get_scaffold_split()
        elif self.split_type == 'pre-define':
            return self.get_pre_define_split()

    def get_random_split(self):
        num_total = len(self.data)
        indices = list(range(num_total))

        train_indices, temp_indices = train_test_split(
                                    indices, 
                                    test_size=self.valid_size + self.test_size, 
                                    random_state=self.args.seed
                                        )
        
        valid_indices, test_indices = train_test_split(
            temp_indices, 
            test_size=self.test_size / (self.valid_size + self.test_size), 
            random_state=self.args.seed
        )
        
        
        train_data = self.data.iloc[train_indices]
        valid_data = self.data.iloc[valid_indices]
        test_data = self.data.iloc[test_indices]
        
        # print(len(train_smiles), len(valid_smiles))

        train_dataset = MoleculeDataLoader(train_data,smiles_to_graph=self.smiles_to_g,additional_features=self.additional_features,args=self.args)
        valid_dataset = MoleculeDataLoader(valid_data,smiles_to_graph=self.smiles_to_g,additional_features=self.additional_features,args=self.args)
        test_dataset = MoleculeDataLoader(test_data,smiles_to_graph=self.smiles_to_g,additional_features=self.additional_features,args=self.args)
        

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            # collate_fn=partial(collate_masking, args=self.args),
            collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False, shuffle=False
        )
        
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, 
            # collate_fn=partial(collate_masking, args=self.args),
            collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False
        )

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            # collate_fn=partial(collate_masking, args=self.args),
            collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False
        )
        
        return train_loader, valid_loader, test_loader

    def generate_scaffold(self,smiles):
        """Generate a scaffold (Bemis-Murcko scaffold) from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)

    def get_scaffold_split(self):
        scaffolds = defaultdict(list)

        # 按照骨架进行分组
        for i, smiles in enumerate(self.data['smiles']):
            scaffold = self.generate_scaffold(smiles)
            scaffolds[scaffold].append(i)

        # 按照骨架组的大小排序
        scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

        train_indices = []
        valid_indices = []
        test_indices = []

        # 按照设置的比例将骨架分配到训练集、验证集和测试集
        for scaffold_set in scaffold_sets:
            if len(train_indices) + len(scaffold_set) <= (1 - self.valid_size - self.test_size) * len(self.data):
                train_indices.extend(scaffold_set)
            elif len(valid_indices) + len(scaffold_set) <= self.valid_size * len(self.data):
                valid_indices.extend(scaffold_set)
            else:
                test_indices.extend(scaffold_set)

        # 分割数据
        train_data = self.data.iloc[train_indices]
        valid_data = self.data.iloc[valid_indices]
        test_data = self.data.iloc[test_indices]

        # 创建数据集和数据加载器
        train_dataset = MoleculeDataLoader(train_data, smiles_to_graph=self.smiles_to_g, additional_features=self.additional_features, args=self.args)
        valid_dataset = MoleculeDataLoader(valid_data, smiles_to_graph=self.smiles_to_g, additional_features=self.additional_features, args=self.args)
        test_dataset = MoleculeDataLoader(test_data, smiles_to_graph=self.smiles_to_g, additional_features=self.additional_features, args=self.args)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False, shuffle=False
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader

    def get_pre_define_split(self):
        """Split dataset based on predefined indices in the dataset."""
        # 建议添加 'split' 列存在性检查
        if 'split' not in self.data.columns:
            raise ValueError("Data must contain 'split' column for pre-defined split")
        
        # 检查分割标签的有效性
        valid_splits = set(['train', 'valid', 'test'])
        if not set(self.data['split'].unique()).issubset(valid_splits):
            raise ValueError("Split column should only contain 'train', 'valid', or 'test'")
        
        # 假设 self.data 中包含一列 'split'，其中标注了 'train', 'valid', 'test'
        
        train_data = self.data[self.data['split'] == 'train'].drop(columns=['split'])
        valid_data = self.data[self.data['split'] == 'valid'].drop(columns=['split'])
        test_data = self.data[self.data['split'] == 'test'].drop(columns=['split'])

        # 创建数据集
        train_dataset = MoleculeDataLoader(train_data, smiles_to_graph=self.smiles_to_g, args=self.args,additional_features=self.additional_features)
        valid_dataset = MoleculeDataLoader(valid_data, smiles_to_graph=self.smiles_to_g, args=self.args,additional_features=self.additional_features)
        test_dataset = MoleculeDataLoader(test_data, smiles_to_graph=self.smiles_to_g, args=self.args,additional_features=self.additional_features)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False, shuffle=False
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader


def split_dataset(args, dataset):
    """Split the dataset for pretrain downstream task
    Parameters
    ----------
    args
        Settings
    dataset
        Dataset instance
    Returns
    -------
    train_set
        Training subset
    val_set
        Validation subset
    test_set
        Test subset
    """
    train_ratio, val_ratio, test_ratio = map(float, args.split_ratio.split(','))
    if args.split == 'scaffold':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args.split == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    else:
        return ValueError("Expect the splitting method to be 'scaffold' or 'random', got {}".format(args.split))

    return train_set, val_set, test_set

def construct_smask(masked_list, N):
    smask_list = []
    for i in range(N):
        if i in masked_list:
            smask_list.append(0)
        else:
            smask_list.append(1)

    return smask_list

def collate_fn(batch):
    """Collate function for supervised pretrain task
    Parameters
    ----------
    batch
        A list of tuples of the form (graph, label)
    Returns
    -------
    bg
        The batched graph
    labels
        The labels
    """
    smiles, graphs, labels = map(list, zip(*batch))
    
    if hasattr(graphs[0], 'molecular_features'):
        molecular_features = []
        for g in graphs:
            molecular_features.append(g.molecular_features)

        molecular_features = torch.stack(molecular_features)
        bg = dgl.batch(graphs)
        setattr(bg, 'molecular_features', molecular_features)
    else:
        bg = dgl.batch(graphs)
    
    # setattr(bg, 'molecular_features', molecular_features)

    labels = [torch.tensor(label) for label in labels]
    labels = torch.stack(labels, dim=0)
    return smiles, bg, labels

def collate_fn_predict(batch):
    smiles, graphs = map(list, zip(*batch))
    if hasattr(graphs[0], 'molecular_features'):
        molecular_features = []
        for g in graphs:
            molecular_features.append(g.molecular_features)

        molecular_features = torch.stack(molecular_features)
        bg = dgl.batch(graphs)
        setattr(bg, 'molecular_features', molecular_features)
    else:
        bg = dgl.batch(graphs)
    
    return smiles, bg

def read_dataset(data_path, target, task):
    smiles_data, labels,split_list = [], [],[]
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                smiles = row['smiles']
                label = row[target]
                split = row['split']
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    split_list.append(split)
                    if task == 'classification':
                        labels.append(int(float(label)))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    # print(len(smiles_data))
    data = pd.DataFrame({'smiles': smiles_data, target: labels, 'split': split_list})
    return data


