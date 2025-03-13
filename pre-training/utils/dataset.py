
from functools import partial
# from .structure_decomposition import get_leaves
from .structure_analyzer import get_structure
from .ifg import get_fgs
import torch

import numpy as np
# import dgl.backend as F
import random
import dgl
from .splitter import ScaffoldSplitter, RandomSplitter
from torch.utils.data import DataLoader
from networkx.algorithms.components import node_connected_component
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
import networkx as nx
from rdkit import Chem
import signal


class TimeoutError(Exception):
    pass


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def get_fragment_indices(mol):
    bonds = mol.GetBonds()
    edges = []
    for bond in bonds:
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    molGraph = nx.Graph(edges)

    BRICS_bonds = list(FindBRICSBonds(mol))
    break_bonds = [b[0] for b in BRICS_bonds]
    break_atoms = [b[0][0] for b in BRICS_bonds] + [b[0][1] for b in BRICS_bonds]
    molGraph.remove_edges_from(break_bonds)

    indices = []
    for atom in break_atoms:
        n = node_connected_component(molGraph, atom)
        if len(n) > 3 and n not in indices:
            indices.append(n)
    indices = set(map(tuple, indices))
    return indices


def get_fragments(mol):
    try:
        with timeout(seconds=2):

            ref_indices = get_fragment_indices(mol)

            frags = list(BRICSDecompose(mol, returnMols=True))
            mol2 = BreakBRICSBonds(mol)

            extra_indices = []
            for i, atom in enumerate(mol2.GetAtoms()):
                if atom.GetAtomicNum() == 0:
                    extra_indices.append(i)
            extra_indices = set(extra_indices)

            frag_mols = []
            frag_indices = []
            for frag in frags:
                indices = mol2.GetSubstructMatches(frag)
                # if len(indices) >= 1:
                #     idx = indices[0]
                #     idx = set(idx) - extra_indices
                #     if len(idx) > 3:
                #         frag_mols.append(frag)
                #         frag_indices.append(idx)
                if len(indices) == 1:
                    idx = indices[0]
                    idx = set(idx) - extra_indices
                    if len(idx) > 3:
                        frag_mols.append(frag)
                        frag_indices.append(idx)
                else:
                    for idx in indices:
                        idx = set(idx) - extra_indices
                        if len(idx) > 3:
                            for ref_idx in ref_indices:
                                if (tuple(idx) == ref_idx) and (idx not in frag_indices):
                                    frag_mols.append(frag)
                                    frag_indices.append(idx)

            return frag_mols, frag_indices
    
    except:
        print('timeout!')
        return [], [set()]



def update_graph(graphs):
        # 计算并存储每个图的全局起始节点,边索引
    cumulative_nodes = 0
    cumulative_edges = 0
    for g in graphs:
        g.ndata['global_index'] = torch.arange(cumulative_nodes, cumulative_nodes + g.number_of_nodes())
        g.edata['global_index'] = torch.arange(cumulative_edges, cumulative_edges + g.number_of_edges())
        cumulative_nodes += g.number_of_nodes()
        cumulative_edges += g.number_of_edges()
        
    # 将每个graph.graph_info中的索引换成全局索引,存储为单独的
    graphs_info_all = []
    for g in graphs:
        g.graph_info['scaffold_atom_idx'] = g.ndata['global_index'][g.graph_info['scaffold_atom_idx']]
        g.graph_info['scaffold_bond_idx'] = g.edata['global_index'][g.graph_info['scaffold_bond_idx']]
        g.graph_info['substituent_atom_idx'] = g.ndata['global_index'][g.graph_info['substituent_atom_idx']]
        g.graph_info['substituent_bond_idx'] = g.edata['global_index'][g.graph_info['substituent_bond_idx']]
        # substituent_fgs_list 是一个列表，列表中的每个元素是一个列表，表示一个功能团的原子序号
        # 通过全局索引将其转换为对应的原子序号
        substituent_fgs_list = []
        for fg in g.graph_info['substituent_fgs_list']:
            fg = [g.ndata['global_index'][i] for i in fg]
            substituent_fgs_list.append(fg)
        g.graph_info['substituent_fgs_list'] = substituent_fgs_list
        graphs_info_all.append(g.graph_info)
    
    # 合并graphs_info_all的数据，按照键值合并
    # 例如，scaffold_atom_idx是一个列表，每个元素是一个原子序号，将这些原子序号合并到一个列表中
    graphs_info_all_dic = {}
    for key in graphs_info_all[0].keys():
        graphs_info_all_dic[key] = []
        for g in graphs_info_all:
            graphs_info_all_dic[key].extend(g[key])
            
    return graphs, graphs_info_all_dic

import torch
import random

def get_mask_nodes_edges(mask_rate, leaves_structure, mask_edge=True, mask_substituent=True):
    """
    Get masked nodes and edges indices in a graph
    """
    
    skeleton_atom_idx = leaves_structure['skeleton']['atom_idx']
    skeleton_bond_idx = leaves_structure['skeleton']['bond_idx']
    
    num_side_chains = len(leaves_structure['side_chains'])
    
    # 计算要掩码的主链节点和边的索引
    masked_sk_nodes_indices = random.sample(skeleton_atom_idx, int(len(skeleton_atom_idx) * mask_rate))
    if mask_edge:   
        masked_sk_edges_indices = random.sample(skeleton_bond_idx, int(len(skeleton_bond_idx) * mask_rate))
    else:
        masked_sk_edges_indices = []
    
    if mask_substituent:
            
        if num_side_chains > 0:
            # 当 side_chains 存在时，计算掩码的数量
            mask_num_side_chains = int(num_side_chains * 0.2)
            if mask_num_side_chains == 0:
                mask_num_side_chains = 1
            masked_side_chains = random.sample(range(num_side_chains), mask_num_side_chains)
            
            masked_sc_nodes_indices = []
            masked_sc_edges_indices = []
            for i in masked_side_chains:
                sc = leaves_structure['side_chains'][i]
                masked_sc_nodes_indices.extend(sc['atom_idx'])
                masked_sc_edges_indices.extend(sc['bond_idx'])

        else:
            # 如果 side_chains 为空，则返回空的 Tensor
            masked_sc_nodes_indices = []
            masked_sc_edges_indices = []
    else:
        masked_sc_nodes_indices = []
        masked_sc_edges_indices = []  

    return masked_sk_nodes_indices, masked_sk_edges_indices, masked_sc_nodes_indices, masked_sc_edges_indices

    
    # mask nodes
    # masked_nodes_indices = torch.LongTensor(
    #     random.sample(bg.graph_info['scaffold_atom_idx'], int(len(bg.graph_info['scaffold_atom_idx']) * mask_rate)))
    # # mask edges
    # if mask_edge:
    #     # masked_edges_indices = mask_edges(bg, masked_nodes_indices) 
    #     # TODO: 验证这个地方，mask edges是根据掩盖的节点来搞的
    #     masked_edges_indices = torch.LongTensor(
    #         random.sample(bg.graph_info['scaffold_bond_idx'], int(len(bg.graph_info['scaffold_bond_idx']) * mask_rate)))
    # else:
    #     masked_edges_indices = torch.LongTensor([])
    
    # if mask_fg:
    #     masked_fgs_nodes_indices = []
    #     random_index = random.sample(range(len(bg.graph_info['substituent_fgs_list'])),int(len(bg.graph_info['substituent_fgs_list']) * mask_rate))
    #     for i in random_index:
    #         masked_fgs_nodes_indices.extend(bg.graph_info['substituent_fgs_list'][i])
    #     masked_fgs_nodes_indices = torch.LongTensor(masked_fgs_nodes_indices)
    #     masked_fgs_edges_indices = mask_edges(bg, masked_fgs_nodes_indices)
        
    #     # masked_nodes_indices extend masked_fgs_nodes_indices
    #     masked_nodes_indices = torch.cat((masked_nodes_indices,masked_fgs_nodes_indices),0)
    #     if mask_edge:
    #         masked_edges_indices = torch.cat((masked_edges_indices,masked_fgs_edges_indices),0)
            
    # return masked_nodes_indices, masked_edges_indices

def mask_graph(graph,masked_nodes_indices,masked_edges_indices):
    """
    Mask nodes and edges in a graph
    """
    # mask nodes, 节点的属性变为0，
    graph.ndata['node'][masked_nodes_indices] = torch.zeros(graph.ndata['node'].shape[1],dtype=torch.float)
    # mask edges
    graph.edata['edge'][masked_edges_indices] = torch.zeros(graph.edata['edge'].shape[1],dtype=torch.float)
    return graph



def collate_masking(batch):
    """
    Collate function to randomly mask nodes/edges
    """
    # gis,gjs,frag_indices,mols,atom_nums = zip(*batch)
    gis, gjs, mols, atom_nums, frag_mols, frag_indices = zip(*batch)
    
    frag_mols = [j for i in frag_mols for j in i]

    gis = dgl.batch(gis)
    gjs = dgl.batch(gjs)

    gis.motif_batch = torch.zeros(gis.number_of_nodes(), dtype=torch.long)
    gjs.motif_batch = torch.zeros(gjs.number_of_nodes(), dtype=torch.long)
    
    # gis.ndata['smask_full'] = torch.ones(gis.number_of_nodes(), dtype=torch.float)
    # gjs.ndata['smask_full'] = torch.ones(gjs.number_of_nodes(), dtype=torch.float)   
    
    curr_indicator = 1
    curr_num = 0
    for N, indices in zip(atom_nums, frag_indices):
        for idx in indices:
            curr_idx = np.array(list(idx)) + curr_num
            gis.motif_batch[curr_idx] = curr_indicator
            gjs.motif_batch[curr_idx] = curr_indicator
            curr_indicator += 1
        curr_num += N
    return gis, gjs, mols, frag_mols

def find_neighbor_edges(g, node_id):
    """Given a node with its graph, return all the edges connected to that node."""
    predecessors = g.predecessors(node_id)
    successors = g.successors(node_id)
    predecessors_edges = g.edge_ids(predecessors, torch.full(predecessors.shape, node_id, dtype=torch.int))
    successors_edges = g.edge_ids(torch.full(successors.shape, node_id, dtype=torch.int), successors)
    return torch.cat((predecessors_edges, successors_edges))


def mask_edges(g, masked_nodes_indices):
    """Given a graph and masked nodes, mask all edges that connected to the masked nodes and return these edge indices."""
    masked_edges_indices = []
    for masked_nodes_index in masked_nodes_indices:
        masked_edges_indices.extend(find_neighbor_edges(g, masked_nodes_index.int()))
    return torch.LongTensor(masked_edges_indices)


# class PretrainDatasetWrapper(object):
#     def __init__(self, batch_size, num_workers, valid_size, data, smiles_to_g):
#         super(object, self).__init__()
#         self.data_path = data
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.valid_size = valid_size
#         self.smiles_to_g = smiles_to_g
        
#     def get_data_loaders(self):
#         num_train = len(self.data)
#         indices = list(range(num_train))
#         np.random.shuffle(indices)
#         split = int(np.floor(self.valid_size * num_train))
#         train_idx, valid_idx = indices[split:], indices[:split]

#         train_data = self.data.iloc[train_idx]
#         valid_data = self.data.iloc[valid_idx]
        
#         train_dataset = PretrainDataset(data=train_data,
#                                       smiles_to_graph=self.smiles_to_g,
#                                       smiles_column='smiles'
#                                     )
#         valid_dataset = PretrainDataset(data=valid_data,
#                                         smiles_to_graph=self.smiles_to_g,
#                                         smiles_column='smiles'
#                                         )
        
#         train_loader = DataLoader(dataset=train_dataset,
#                                   batch_size=self.batch_size,
#                                   shuffle=True,
#                                   drop_last=True,
#                                   num_workers=self.num_workers,
#                                   collate_fn=collate_fn)
        
#         valid_loader = DataLoader(dataset=valid_dataset,
#                                     batch_size=self.batch_size,
#                                     drop_last=True,
#                                     num_workers=self.num_workers,
#                                     collate_fn=collate_fn)
        
#         return train_loader,valid_loader
    
def collate_fn(batch):
    graphs = [graph for graph in batch]
    bg = dgl.batch(graphs)
    return bg




def construct_smask(masked_list, N):
    smask_list = []
    for i in range(N):
        if i in masked_list:
            smask_list.append(0)
        else:
            smask_list.append(1)

    return smask_list

class MoleculeDataLoader(object):
    """
    adapted from https://lifesci.dgl.ai/_modules/dgllife/data/csv_dataset.html#MoleculeCSVDataset
    used for pretrain_masking(task=masking) and pretrain_supervised(task=supervised) task.
    """

    def __init__(self, data, smiles_to_graph,args):
        self.smiles = data
        self.smiles_to_graph = smiles_to_graph
        self.args = args

    def __getitem__(self, item):
        s = self.smiles[item]

        mol = Chem.MolFromSmiles(s)
        
        # add some information to the graph
        leaves_structure = get_structure(smiles=s)
        
        masked_sk_nodes_indices_i, masked_sk_edges_indices_i, masked_sc_nodes_indices_i, masked_sc_edges_indices_i = get_mask_nodes_edges(self.args.mask_rate,leaves_structure)
        masked_sk_nodes_indices_j, masked_sk_edges_indices_j, masked_sc_nodes_indices_j, masked_sc_edges_indices_j = get_mask_nodes_edges(self.args.mask_rate,leaves_structure)
        
        masked_nodes_indices_i = masked_sk_nodes_indices_i + masked_sc_nodes_indices_i
        masked_edges_indices_i = masked_sk_edges_indices_i + masked_sc_edges_indices_i
        
        masked_nodes_indices_j = masked_sk_nodes_indices_j + masked_sc_nodes_indices_j
        masked_edges_indices_j = masked_sk_edges_indices_j + masked_sc_edges_indices_j
        
        graph_i = self.smiles_to_graph(s,mask_nodes_indices=masked_nodes_indices_i,mask_edges_indices=masked_edges_indices_i)
        graph_j = self.smiles_to_graph(s,mask_nodes_indices=masked_nodes_indices_j,mask_edges_indices=masked_edges_indices_j)
        

        
        # side chains drop
        # smask包含所有的要掩码的节点，sknodes + sc_nodes, 要在后续的readout地方应用。
        # smask_i = torch.tensor(construct_smask(torch.cat((masked_sk_nodes_indices_i,masked_sc_nodes_indices_i)), graph_i.number_of_nodes())).float()
        # smask_j = torch.tensor(construct_smask(torch.cat((masked_sk_nodes_indices_j, masked_sc_nodes_indices_j)), graph_j.number_of_nodes())).float()
        
        # graph_i.ndata['smask']  = smask_i
        # graph_j.ndata['smask']  = smask_j
        
        # drop sc edges
        # graph_i.remove_edges(masked_sc_edges_indices_i)
        # graph_j.remove_edges(masked_sc_edges_indices_j)
        
        frag_mols, frag_indices = get_fragments(mol)

        N = mol.GetNumAtoms()
        
        # return g_i,g_j, frag_indices,mol, N
        return graph_i,graph_j,mol, N, frag_mols, frag_indices


    def __len__(self):
        return len(self.smiles)


class MoleculeDataset(object):
    def __init__(self, data,smiles_to_graph,args):
        super(object, self).__init__()
        # self.data = data
        self.smiles = data
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.valid_size = args.valid_size
        self.smiles_to_g = smiles_to_graph
        self.args = args
        
    def get_data_loaders(self):
        
        num_train = len(self.smiles)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_smiles = self.smiles.iloc[train_idx]['smiles'].tolist()
        valid_smiles = self.smiles.iloc[valid_idx]['smiles'].tolist()
        
        # print(len(train_smiles), len(valid_smiles))

        train_dataset = MoleculeDataLoader(train_smiles,smiles_to_graph=self.smiles_to_g,args=self.args)
        valid_dataset = MoleculeDataLoader(valid_smiles,smiles_to_graph=self.smiles_to_g,args=self.args)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            # collate_fn=partial(collate_masking, args=self.args),
            collate_fn=collate_masking,
            pin_memory=True,
            # prefetch_factor=1,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )
        
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, 
            # collate_fn=partial(collate_masking, args=self.args),
            collate_fn=collate_masking,
            pin_memory=True,
            # prefetch_factor=2,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )

        return train_loader, valid_loader


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


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally a binary
        mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels.
    """
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks





