#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   structure_decomposition.py
@Time    :   2024/04/20 17:33:37
@Author  :   Zhang Qian
@Contact :   zhangqian.allen@gmail.com
@License :   Copyright (c) 2024 by Zhang Qian, All Rights Reserved. 
@Desc    :   None
"""

# here put the import lib
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import BRICS
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from multiprocessing import Pool
import networkx as nx

"""
1. judge if a fused rings system is exist in a molecule
2. find the fused rings system as the scaffold
3. if not exist, find the Murcko scaffold
4. find the link bonds between the scaffold and the substituent
5. find substituent

return {'substituent':{'atom_idx':[[],[]],'bond_idx':[[],[]]},
        'scaffold':{'atom_idx':[],'bond_idx':[]},
        'fused_rings_scaffold':bool,
        'murcko_scaffold':bool,
        }

"""


def get_fused_rings_atom_idx(mol):
    """
    Get the atom indices of the fused rings in a molecule.

    Parameters:
    mol (Chem.Mol): The input molecule.

    Returns:
    set: A set of atom indices that belong to the fused rings.
    int: number of fused rings
    """
    ssr = Chem.GetSymmSSSR(mol)  
    fused_rings = []
    for i in range(len(ssr)):
        for j in range(i + 1, len(ssr)):
            if len(set(ssr[i]) & set(ssr[j])) == 2:  
                if set(ssr[i]).issubset(set(ssr[j])) or set(ssr[j]).issubset(set(ssr[i])):
                    continue  
                if ssr[i] not in fused_rings:
                    fused_rings.append(ssr[i])
                if ssr[j] not in fused_rings:
                    fused_rings.append(ssr[j])
                    
    fused_rings_atom_idx = []
    for ring in fused_rings:
        fused_rings_atom_idx.extend(ring)
        
    return set(fused_rings_atom_idx),len(fused_rings)

def get_fused_rings_atom_idx_nx(mol):
    """
    Get the atom indices of the fused rings in a molecule.

    Parameters:
    mol (Chem.Mol): The input molecule.

    Returns:
    """
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        
    g = nx.Graph()
    for i in range(len(ssr)):
        for j in range(i + 1, len(ssr)):
            if any(atom in ssr[i] for atom in ssr[j]):  # 如果两个环有共享原子
                g.add_edge(i, j)
    
    components = list(nx.connected_components(g))
    
    merged_rings = []
    for component in components:
        merged = set()
        for idx in component:
            merged.update(ssr[idx])
        merged_rings.append(sorted(merged))
    
    # fused_rings_atom_idx = []
    # for ring in merged_rings:
    #     fused_rings_atom_idx.extend(ring)
        
    return merged_rings
 
 
def find_paths(graph, start, end, max_depth=10):
    """
    迭代深入查找路径，而不是递归。注意。max_depth可能需要进一步调节
    """
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        if len(path) > max_depth:
            continue
        for next_node in set(graph[vertex]) - set(path):
            if next_node == end:
                yield path + [next_node]
            else:
                stack.append((next_node, path + [next_node]))

def find_bridge_atoms_between_fused_systems(molecule, systems):
    # 构建分子的图表示
    g = nx.Graph()
    num_atoms = molecule.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if molecule.GetBondBetweenAtoms(i, j):
                g.add_edge(i, j)
    
    # 转换NetworkX图到字典表示，以便进行递归路径查找
    graph = {node: list(adjacent) for node, adjacent in nx.to_dict_of_lists(g).items()}

    # 确保每个系统是一个集合，方便进行集合操作
    systems = [set(system) for system in systems]
    
    # 找到所有稠环系统中的原子
    fused_atoms = set().union(*systems)

    # 初始化桥接原子集合
    bridge_atoms = set()

    # 检查所有系统间的连接
    for i, system1 in enumerate(systems):
        for system2 in systems[i+1:]:  # 避免自身与自身比较，仅比较不同系统
            # 寻找两个稠环系统之间的所有连接路径
            for atom1 in system1:
                for atom2 in system2:
                    for path in find_paths(graph, atom1, atom2):
                        # 选择路径上不属于任何稠环系统的原子
                        bridge_atoms.update([p for p in path if p not in fused_atoms])

    # 返回桥接原子集合
    return list(bridge_atoms)
 
 
def get_substituent_and_fused_rings_scaffold(mol,fused_rings):

    substituent_and_scaffold = dict()
    
    
    if len(fused_rings) == 1:
        # get fused rings scaffold
        fused_rings_atom_idx = list(fused_rings)[0]
        
        substituent_and_scaffold['fused_rings_scaffold'] = True
        substituent_and_scaffold['murcko_scaffold'] = False
        substituent_and_scaffold['scaffold'] = {'atom_idx':list(fused_rings_atom_idx),'bond_idx':find_bond_index(mol, fused_rings_atom_idx)}
        
        substituent_atom_idx = [x for x in range(mol.GetNumAtoms()) if x not in fused_rings_atom_idx]
        substituent_and_scaffold['substituent'] = {'atom_idx':substituent_atom_idx,
                                                'bond_idx':find_bond_index(mol, substituent_atom_idx)}
        
        
    elif len(fused_rings)>1:
        substituent_and_scaffold['fused_rings_scaffold'] = True
        substituent_and_scaffold['murcko_scaffold'] = False
        
        bridge_atoms = find_bridge_atoms_between_fused_systems(mol, fused_rings)
        
        all_fused_rings_atom_idx = []
        for fused_rings_atom_idx in fused_rings:
            all_fused_rings_atom_idx.extend(fused_rings_atom_idx)
        all_fused_rings_atom_idx.extend(bridge_atoms)
                
        substituent_atom_idx = [x for x in range(mol.GetNumAtoms()) if x not in all_fused_rings_atom_idx]
        
        substituent_and_scaffold['scaffold'] = {'atom_idx':all_fused_rings_atom_idx,'bond_idx':find_bond_index(mol, all_fused_rings_atom_idx)}
        
        substituent_and_scaffold['substituent'] = {'atom_idx':substituent_atom_idx,'bond_idx':find_bond_index(mol, substituent_atom_idx)}
        
    return substituent_and_scaffold

def find_murcko_link_bond(mol):
    """
    Finds the link bonds in a molecule that connect to the Murcko scaffold.

    Args:
        mol (Chem.rdchem.Mol): The input molecule.

    Returns:
        tuple: A tuple containing the list of link bonds and the scaffold index.
            The list of link bonds is represented as a list of tuples, where each tuple
            contains the indices of the atoms connected by the link bond.
            The scaffold index is represented as a list of atom indices that belong to the Murcko scaffold.
    """
    core = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_index = mol.GetSubstructMatch(core)
    link_bond_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        link_score = 0
        if u in scaffold_index:
            link_score += 1
        if v in scaffold_index:
            link_score += 1
        if link_score == 1:
            link_bond_list.append([u, v])
    return link_bond_list, list(scaffold_index)

def find_bond_index(mol, atom_list):
    """
    Find the indices of bonds in a molecule that connect atoms from the given atom_list.

    Args:
        mol (Chem.Mol): The molecule object.
        atom_list (list): List of atom indices.

    Returns:
        list: List of bond indices.

    """
    bond_indices = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_list and bond.GetEndAtomIdx() in atom_list:
            bond_indices.append(bond.GetIdx())
    
    return bond_indices

def get_substituent_and_murcko(m):

    """
    {'substituent':{'atom_idx':[[],[]],'bond_idx':[[],[]]},
    'scaffold':{'atom_idx':[],'bond_idx':[]}

    """
    
    substituent_and_scaffold = dict()
    
    substituent_dir = dict()
    scaffold_dir = dict()
    
    
    # return murcko_link_bond
    all_murcko_bond,scaffold_index = find_murcko_link_bond(m)
    scaffold_index.sort()
    scaffold_dir['atom_idx'] = scaffold_index
    
    
    scaffold_bonds = find_bond_index(m, scaffold_index)
    scaffold_bonds.sort()
    scaffold_dir['bond_idx'] = scaffold_bonds    
    
    
    # return atom in all_murcko_bond
    all_murcko_atom = []
    for murcko_bond in all_murcko_bond:
        all_murcko_atom = list(set(all_murcko_atom + murcko_bond))

    if len(all_murcko_atom) > 0:
        # return all break atom (the break atoms did'n appear in the same substructure)
        all_break_atom = dict()
        for murcko_atom in all_murcko_atom:
            murcko_break_atom = []
            for murcko_bond in all_murcko_bond:
                if murcko_atom in murcko_bond:
                    murcko_break_atom += list(set(murcko_bond))
            murcko_break_atom = [x for x in murcko_break_atom if x != murcko_atom]
            all_break_atom[murcko_atom] = murcko_break_atom

        # substituent_idx = dict()
        substituent_idx_all = []
        used_atom = []
        for initial_atom_idx, break_atoms_idx in all_break_atom.items():
            if initial_atom_idx not in used_atom:
                neighbor_idx = [initial_atom_idx]
                substituent_idx_i = neighbor_idx
                begin_atom_idx_list = [initial_atom_idx]
                while len(neighbor_idx) != 0:
                    for idx in begin_atom_idx_list:
                        initial_atom = m.GetAtomWithIdx(idx)
                        neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                       initial_atom.GetNeighbors()]
                        exlude_idx = all_break_atom[initial_atom_idx] + substituent_idx_i
                        if idx in all_break_atom.keys():
                            exlude_idx = all_break_atom[initial_atom_idx] + substituent_idx_i + all_break_atom[idx]
                        neighbor_idx = [x for x in neighbor_idx if x not in exlude_idx]
                        substituent_idx_i += neighbor_idx
                        begin_atom_idx_list += neighbor_idx
                    begin_atom_idx_list = [x for x in begin_atom_idx_list if x not in substituent_idx_i]
                substituent_idx_i.sort()
                if substituent_idx_i != scaffold_index:
                    substituent_idx_all.append(substituent_idx_i)
                    # substituent_idx[initial_atom_idx] = substituent_idx_i
                used_atom += substituent_idx_i
            else:
                pass
        
        # substituent_atom_idx_all = [x for x in substituent_idx.values()]
        substituent_atom_idx_all = []
        for substituent in substituent_idx_all:
            substituent.sort()
            substituent_atom_idx_all.extend(substituent)
        substituent_dir['atom_idx'] = substituent_atom_idx_all
        
        
        substituent_bonds_all = []
        for substituent in substituent_idx_all:
            substituent_bonds = find_bond_index(m, substituent)
            substituent_bonds.sort()
            substituent_bonds_all.extend(substituent_bonds)
            
        
        substituent_dir['bond_idx'] = substituent_bonds_all
        
    else:
        substituent_dir['atom_idx'] = [x for x in range(m.GetNumAtoms())]
        substituent_dir['bond_idx'] = [x for x in range(m.GetNumBonds())]
        
    substituent_and_scaffold['substituent'] = substituent_dir
    substituent_and_scaffold['scaffold'] = scaffold_dir
    substituent_and_scaffold['murcko_scaffold'] = True
    substituent_and_scaffold['fused_rings_scaffold'] = False
    
    return substituent_and_scaffold
    


def get_leaves(smiles):
    """
    Get the leaves of a molecule.
    """
    mol = MolFromSmiles(smiles)
    
    fused_rings = get_fused_rings_atom_idx_nx(mol)

    if len(fused_rings) > 0:
        substituent_and_scaffold = get_substituent_and_fused_rings_scaffold(mol,fused_rings)        
        # get substituent and scaffold
    else:
        substituent_and_scaffold = get_substituent_and_murcko(mol)
    
    # append smiles
    # substituent_and_scaffold['smiles'] = smiles
    
    
    return substituent_and_scaffold

def get_leaves_multiprocess(smiles_list,num_workers=8):
    
    if num_workers == 1:
        results = []
        for smiles in smiles_list:
            results.append(get_leaves(smiles))
    else:
        with Pool(num_workers) as p:
            results = p.map(get_leaves, smiles_list)
    
    return results



if __name__=='__main__':
    smiles = "O=C1OC2=CC3=C(OC(=O)C3=C4C(=O)N(C5=CC(Br)=CC=C45)CCCC(CCCCCCCCCCCCCCCCCC)CCCCCCCCCCCCCCCCCC)C=C2C1=C6C(=O)N(C=7C=C(Br)C=CC67)CCCC(CCCCCCCCCCCCCCCCCC)CCCCCCCCCCCCCCCCCC"
    print(get_leaves(smiles))
    
    