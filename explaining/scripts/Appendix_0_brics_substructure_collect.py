from multiprocessing import Pool
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from build_data import return_brics_leaf_structure
# 全局变量，用于存储预处理后的 attribution 数据
smiles_to_attribution = {}

def init_pool(data):
    """进程池初始化函数，用于设置全局变量"""
    global smiles_to_attribution
    smiles_to_attribution = data

def extract_substructure(smiles, atom_indices, return_format='smiles'):
    """
    从给定的SMILES字符串中提取指定原子索引的子结构。
    
    参数：
    - smiles (str): 输入的SMILES字符串。
    - atom_indices (list of int): 要提取的子结构的原子索引列表（基于0）。
    - return_format (str): 返回格式，可以是'smiles'或'smarts'。默认为'smiles'。
    
    返回：
    - str: 提取的子结构的SMILES或SMARTS字符串。如果提取失败，返回None。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("无效的SMILES字符串。")
        return None

    num_atoms = mol.GetNumAtoms()
    for idx in atom_indices:
        if idx < 0 or idx >= num_atoms:
            print(f"原子索引 {idx} 超出范围。分子中有 {num_atoms} 个原子。")
            return None

    atom_indices_set = set(atom_indices)
    bonds = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in atom_indices_set and end in atom_indices_set:
            bonds.append(bond.GetIdx())

    submol = Chem.RWMol()
    mapping = {}
    for new_idx, old_idx in enumerate(atom_indices):
        atom = mol.GetAtomWithIdx(old_idx)
        mapping[old_idx] = new_idx
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetIsAromatic(atom.GetIsAromatic())
        submol.AddAtom(new_atom)

    for bond in bonds:
        b = mol.GetBondWithIdx(bond)
        submol.AddBond(mapping[b.GetBeginAtomIdx()], mapping[b.GetEndAtomIdx()], b.GetBondType())
        if b.GetIsAromatic():
            submol.GetBondWithIdx(submol.GetNumBonds()-1).SetIsAromatic(True)

    submol = submol.GetMol()

    if return_format == 'smiles':
        return Chem.MolToSmiles(submol, isomericSmiles=True)
    elif return_format == 'smarts':
        return Chem.MolToSmarts(submol)
    else:
        print("无效的返回格式。请选择'smiles'或'smarts'。")
        return None

def process_single_smiles(smiles):
    """
    处理单个SMILES字符串，提取对应的 attribution 数据。
    
    参数：
    - smiles (str): 输入的SMILES字符串。
    
    返回：
    - list: 对应的 attribution 数据列表。如果没有匹配，返回空列表。
    """
    attribution = smiles_to_attribution.get(smiles, [])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol_num = mol.GetNumAtoms()
    return attribution[mol_num:]

def main():
    property_name  = 'hr'
    sub_type = 'brics'
    
    # 读取原始数据
    data = pd.read_csv(f'../data/origin_data/{property_name}.csv')
    smiles_list = data['smiles'].tolist()
    brics_information = []
    smask_npy = np.load('../data/graph_data/hr_smask_for_brics.npy',allow_pickle=True)
    
    
    # 处理 BRICS 子结构
    # for input_smiles in tqdm(smiles_list, desc='Processing BRICS'):
    #     substructure_dir = return_brics_leaf_structure(input_smiles)
    #     for _, substructure in substructure_dir['substructure'].items():
    #         is_valid = 0
    #         sub_smiles = extract_substructure(input_smiles, substructure, return_format='smiles')
    #         sub_smarts = extract_substructure(input_smiles, substructure, return_format='smarts')
    #         if sub_smiles is not None and sub_smiles != input_smiles:
    #             is_valid = 1
    #         brics_information.append([input_smiles, substructure, sub_smiles, sub_smarts, is_valid])
    
    # brics_information_df = pd.DataFrame(brics_information, columns=['smiles','substructure_atom_indices','substructure_smiles','substructure_smarts','is_valid'])
    # brics_information_df.to_csv('../Figure-4a-data/brics-scaffold/brics_information.csv', index=False)
    
    # 处理 Attribution 数据
    for i in [3]:
        attribution_data_path = f'../prediction/attribution/{property_name}_{sub_type}_cv_{i}_attribution_summary.csv'
        attribution_data = pd.read_csv(attribution_data_path)
        # 将attribution为NaN的填充为0
        attribution_data['attribution'] = attribution_data['attribution'].fillna(0)
        
        for input_smiles in tqdm(smiles_list,desc=f'cv_{i}'):
            num_atoms = Chem.MolFromSmiles(input_smiles).GetNumAtoms()
            substructure_dir = return_brics_leaf_structure(input_smiles)
            attribution_i = list(attribution_data[attribution_data['smiles'] == input_smiles]['attribution'])
            attribution_i = attribution_i[num_atoms:num_atoms+len(substructure_dir['substructure'])]

            if len(substructure_dir['substructure']) != len(attribution_i):
                print(f'{input_smiles} 的 BRICS 子结构数量与 attribution 数量不一致')
                print(len(substructure_dir['substructure']), len(attribution_i))
                break
            
            for num_substructure, (_,substructure) in enumerate(substructure_dir['substructure'].items()):
                is_valid = 0
                sub_smiles = extract_substructure(input_smiles, substructure, return_format='smiles')
                sub_smarts = extract_substructure(input_smiles, substructure, return_format='smarts')
                if sub_smiles is not None and sub_smiles != input_smiles:
                    is_valid = 1
                
                # print(num_substructure)
                # print(attribution_i)

                attribution = attribution_i[int(num_substructure)]
                brics_information.append([input_smiles, substructure, sub_smiles, sub_smarts, is_valid, attribution])
        
        brics_information_df = pd.DataFrame(brics_information, columns=['smiles','substructure_atom_indices','substructure_smiles','substructure_smarts','is_valid','attribution'])
        brics_information_df.to_csv(f'../Figure-4a-data/brics-scaffold/brics_information_attribution_cv_{i}.csv', index=False)
        
        # # 预处理 attribution_data，转换为字典以加快查找速度
        # smiles_to_attr = attribution_data.groupby('smiles')['attribution'].apply(list).to_dict()

        
        # # 创建进程池并行处理
        # with Pool(processes=20, initializer=init_pool, initargs=(smiles_to_attr,)) as pool:
        #     attribution_list = []
        #     for result in tqdm(pool.imap(process_single_smiles, data['smiles']), total=len(data['smiles']), desc=f'Processing Attribution cv_{i}'):
        #         attribution_list.extend(result)
        
        # print(f"Attribution 列表长度: {len(attribution_list)}, BRICS 信息长度: {len(brics_information_df)}")
        # assert len(attribution_list) == len(brics_information_df), 'attribution_list 和 brics_information_df 长度不一致'
        # brics_information_df['attribution'] = attribution_list
        # brics_information_df.to_csv(f'../Figure-4a-data/brics-scaffold/brics_information_attribution_cv_{i}.csv', index=False)

if __name__ == "__main__":
    main()
