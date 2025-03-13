import os
import numpy as np
import random
import math
from rdkit import Chem
from collections import defaultdict
from copy import deepcopy
import pandas as pd  # 导入 pandas 库
import pickle
from joblib import Parallel, delayed
from rdkit import RDLogger
import multiprocessing

class SubstructureNode:
    def __init__(self, atom_indices, parent=None):
        self.atom_indices = atom_indices  # 当前子结构的原子索引集合
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0
        self.untried_actions = None  # 未尝试的动作列表
        self.is_fully_expanded_flag = False

    def is_fully_expanded(self):
        return self.is_fully_expanded_flag

    def expand(self, molecule, atom_contributions):
        if self.untried_actions is None:
            self.untried_actions = self.generate_possible_actions(molecule)
        if not self.untried_actions:
            self.is_fully_expanded_flag = True
            return None
        action = self.untried_actions.pop()
        new_atom_indices = self.perform_action(action, molecule)
        if new_atom_indices is not None:
            child_node = SubstructureNode(new_atom_indices, parent=self)
            self.children.append(child_node)
            return child_node
        else:
            return self.expand(molecule, atom_contributions)

    def generate_possible_actions(self, molecule):
        actions = []
        current_atoms = self.atom_indices
        possible_additions = set()
        for idx in current_atoms:
            atom = molecule.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx not in current_atoms:
                    possible_additions.add(n_idx)
        for idx in possible_additions:
            actions.append(('add', idx))
        if len(current_atoms) > 1:
            for idx in current_atoms:
                actions.append(('remove', idx))
        return actions

    def perform_action(self, action, molecule):
        action_type, idx = action
        new_atom_indices = set(self.atom_indices)
        if action_type == 'add':
            new_atom_indices.add(idx)
        elif action_type == 'remove':
            new_atom_indices.remove(idx)
            if not self.is_connected_subgraph(new_atom_indices, molecule):
                return None
        return new_atom_indices

    def is_connected_subgraph(self, atom_indices, molecule):
        if not atom_indices:
            return False
        visited = set()
        to_visit = set([next(iter(atom_indices))])
        while to_visit:
            idx = to_visit.pop()
            visited.add(idx)
            atom = molecule.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx in atom_indices and n_idx not in visited:
                    to_visit.add(n_idx)
        return visited == atom_indices

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.reward / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def is_terminal(self):
        return False  # 可以根据条件定义终止

    def rollout(self, molecule, atom_contributions):
        current_atom_indices = set(self.atom_indices)
        depth = 0
        max_depth = 5
        while depth < max_depth:
            actions = self.generate_possible_actions(molecule)
            if not actions:
                break
            action = random.choice(actions)
            new_atom_indices = self.perform_action(action, molecule)
            if new_atom_indices is not None and new_atom_indices != current_atom_indices:
                current_atom_indices = new_atom_indices
            depth += 1
        reward = self.compute_reward(current_atom_indices, atom_contributions)
        return reward

    def compute_reward(self, atom_indices, atom_contributions):
        total_contribution = sum(atom_contributions.get(idx, 0) for idx in atom_indices)
        return -total_contribution  # 奖励为负贡献，即降低重组能

def mcts(root, molecule, atom_contributions, iterations):
    for _ in range(iterations):
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        if not node.is_terminal():
            child = node.expand(molecule, atom_contributions)
            if child is not None:
                node = child
        reward = node.rollout(molecule, atom_contributions)
        backpropagate(node, reward)
    return extract_best_substructures(root, molecule)

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.reward += reward
        node = node.parent

def extract_best_substructures(root, molecule):
    """
    提取最佳子结构，返回SMARTS表示和对应的原子索引
    """
    best_nodes = []
    stack = [root]
    visited_substructures = set()
    
    while stack:
        node = stack.pop()
        atom_indices_tuple = tuple(sorted(node.atom_indices))
        if atom_indices_tuple in visited_substructures:
            continue
            
        visited_substructures.add(atom_indices_tuple)
        if node.children:
            stack.extend(node.children)
            
        if node.visits > 0:
            avg_reward = node.reward / node.visits
            submol = Chem.PathToSubmol(molecule, list(node.atom_indices))
            smarts = Chem.MolToSmarts(submol)
            # 保存SMARTS和原子索引
            best_nodes.append({
                'smarts': smarts,
                'atom_indices': list(sorted(node.atom_indices)),
                'reward': avg_reward
            })
            
    # 按reward排序
    best_nodes.sort(key=lambda x: x['reward'], reverse=True)
    return best_nodes[:10]

def save_results(results, save_path):
    """
    保存结果，包括SMARTS和原子索引
    """
    # 保存详细结果到文本文件
    with open(os.path.join(save_path, 'molecule_results.txt'), 'w') as f:
        for result in results:
            f.write(f"SMILES: {result['smiles']}\n")
            for substructure in result['best_substructures']:
                f.write(f"  SMARTS: {substructure['smarts']}\n")
                f.write(f"  Atom Indices: {substructure['atom_indices']}\n")
                f.write(f"  Reward: {substructure['reward']}\n")
                f.write("\n")
            f.write("-" * 50 + "\n")

def analyze_global_results(results, substructure_to_molecules, save_path):
    """
    全局分析结果，按子结构统计
    """
    substructure_stats = {}
    
    # 统计每个子结构的信息
    for result in results:
        smiles = result['smiles']
        for substructure in result['best_substructures']:
            smarts = substructure['smarts']
            if smarts not in substructure_stats:
                substructure_stats[smarts] = {
                    'count': 0,
                    'molecule_list': set(),
                    'atom_indices_dict': {}
                }
            substructure_stats[smarts]['count'] += 1
            substructure_stats[smarts]['molecule_list'].add(smiles)
            substructure_stats[smarts]['atom_indices_dict'][smiles] = substructure['atom_indices']
    
    # 转换为DataFrame格式
    substructure_data = []
    for smarts, stats in substructure_stats.items():
        substructure_data.append({
            'Substructure': smarts,
            'Count': stats['count'],
            'Molecule List': ', '.join(sorted(stats['molecule_list'])),  # 使用逗号分隔
            'Atom_Indices_Dict': stats['atom_indices_dict']
        })
    
    # 创建DataFrame并按Count降序排序
    df = pd.DataFrame(substructure_data)
    df = df.sort_values('Count', ascending=False)
    
    # 保存为CSV
    df.to_csv(os.path.join(save_path, 'substructure_analysis.csv'), index=False)
    
    # 保存完整数据结构
    with open(os.path.join(save_path, 'substructure_analysis.pkl'), 'wb') as f:
        pickle.dump(substructure_data, f)
    
    print("全局分析结果已保存到 substructure_analysis.csv 和 substructure_analysis.pkl")

# 新增的函数，用于处理单个分子
def process_single_molecule(data, iterations):
    """
    处理单个分子，返回包含原子索引的结果
    """
    smiles = data['smiles']
    atom_contributions = data['atom_contributions']
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
        
    min_contrib_atom = min(atom_contributions, key=atom_contributions.get)
    initial_atom_indices = set([min_contrib_atom])
    root_node = SubstructureNode(initial_atom_indices)
    best_substructures = mcts(root_node, molecule, atom_contributions, iterations)
    
    return {
        'smiles': smiles,
        'best_substructures': best_substructures
    }

# 修改后的主函数，使用并行处理
def process_molecules(molecules_data, iterations=1000, save_path='./results_mcts'):
    global_results = []  # 用于存储所有分子的结果
    substructure_to_molecules = defaultdict(set)  # 用于存储每个子结构在哪些分子中出现

    # 获取 CPU 核心数量
    num_cores = multiprocessing.cpu_count()
    print(f"使用 {num_cores} 个 CPU 核心进行并行处理。")

    # 使用 joblib 进行并行处理
    results = Parallel(n_jobs=num_cores)(
        delayed(process_single_molecule)(data, iterations) for data in molecules_data)

    # 收集结果
    for result in results:
        if result is None:
            continue  # 跳过无法处理的分子
        global_results.append(result)
        smiles = result['smiles']
        best_substructures = result['best_substructures']
        # 将子结构与其所在的分子进行关联
        for substructure in best_substructures:
            substructure_to_molecules[substructure['smarts']].add(smiles)
    
    # 保存结果为文件或者其他存储格式
    save_results(global_results, save_path)
    
    # 全局统计分析并保存为 CSV
    analyze_global_results(global_results, substructure_to_molecules, save_path)

# 示例使用
if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')  # 禁用RDKit警告信息

    for attribution_name in ['delta_est','hl','s0s1','s0t1','hr']:
        # 示例分子数据
        attribution_all_list = pickle.load(open(f'./{attribution_name}/best_attribution.pkl', 'rb'))
        molecules_data = attribution_all_list
        save_path = f'./results_mcts_iter2000/{attribution_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        iterations = 2000
        process_molecules(molecules_data, iterations, save_path)
